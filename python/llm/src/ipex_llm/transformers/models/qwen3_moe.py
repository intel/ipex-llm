import torch
from typing import Optional, List, Tuple
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel, Qwen3MoeAttention

from ipex_llm.transformers.kv import DynamicNormalCache
from ipex_llm.transformers.models.common import merge_qkv_base
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.models.utils import use_fuse_moe
from ipex_llm.transformers.models.utils import make_cache_contiguous_inplaced


def merge_qkv(module: torch.nn.Module):
    merge_qkv_base(module, Qwen3MoeAttention)


def qwen3_moe_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> MoeModelOutputWithPast:
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    use_cache = True if device.type == "xpu" else use_cache
    if use_cache and not isinstance(past_key_values, DynamicNormalCache):
        past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)

    return Qwen3MoeModel.forward(
        self=self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        **flash_attn_kwargs,
    )


def qwen3_moe_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, -1, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.config.num_attention_heads,
                                                        self.config.num_key_value_heads,
                                                        self.config.num_key_value_heads], dim=1)
    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    cos, sin = position_embeddings
    if device.type == "xpu":
        import xe_addons
        make_cache_contiguous_inplaced(cos, sin)
        xe_addons.rotary_half_with_cache_inplaced(query_states, key_states, cos, sin)
    else:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, None)
    attn_weights = None
    attn_output = scaled_dot_product_attention(
        query_states, key_states, value_states,
        attention_mask, q_len == key_states.size(2), self.scaling
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def qwen3_moe_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    if router_logits.device == "xpu":
        import xe_addons
        selected_experts, routing_weights = xe_addons.moe_softmax_topk(
            router_logits, self.top_k, self.norm_topk_prob
        )
    else:
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

    if selected_experts.size(0) == 1:
        if use_fuse_moe(hidden_states, self.experts[0].down_proj.qtype):
            if getattr(self, "gates", None) is None:
                gate_addrs = [expert.gate_proj.weight.data_ptr() for expert in self.experts]
                up_addrs = [expert.up_proj.weight.data_ptr() for expert in self.experts]
                down_addrs = [expert.down_proj.weight.data_ptr() for expert in self.experts]
                gates = torch.tensor(gate_addrs, dtype=torch.uint64, device=hidden_states.device)
                ups = torch.tensor(up_addrs, dtype=torch.uint64, device=hidden_states.device)
                downs = torch.tensor(down_addrs, dtype=torch.uint64, device=hidden_states.device)
                self.register_buffer("gates", gates, persistent=False)
                self.register_buffer("ups", ups, persistent=False)
                self.register_buffer("downs", downs, persistent=False)

            import xe_linear
            final_hidden_states = xe_linear.moe_forward_vec(
                hidden_states, selected_experts, routing_weights, self.gates, self.ups, self.downs,
                hidden_states.size(-1), self.experts[0].intermediate_size,
                self.experts[0].down_proj.qtype
            )
        else:
            idxs = selected_experts.flatten().tolist()
            outputs = []
            for i in idxs:
                expert = self.experts[i]
                expert_out = expert(hidden_states)
                outputs.append(expert_out)
            outs = torch.cat(outputs, dim=0)
            reshaped_topk_weight = routing_weights.squeeze(0).unsqueeze(-1)
            final_hidden_states = (outs * reshaped_topk_weight).sum(dim=0, keepdim=True)
    else:
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts,
                                                  num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits
