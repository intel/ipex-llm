#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some parts of this file is adapted from
# https://github.com/huggingface/transformers/blob/3a1ead0aabed473eafe527915eea8c197d424356/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py
# which is licensed under Apache License 2.0

import math
import torch
from typing import Optional, Tuple, List, Union
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAttention
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_rotary_pos_emb
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_rotary_pos_emb_vision
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_multimodal_rotary_pos_emb

from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers.kv import DynamicNormalCache
from ipex_llm.transformers.models.common import merge_qkv_base
from ipex_llm.transformers.models.common import attention_softmax
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.models.utils import use_sdp_non_causal


def merge_qkv(module: torch.nn.Module):
    merge_qkv_base(module, Qwen2_5OmniAttention)


def qwen2_5_omni_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor]=None,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor]=None,
):
    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_key_value_heads,
                                                        self.num_key_value_heads], dim=1)

    cos, sin = position_embeddings
    if query_states.device.type == "xpu":
        import xe_addons
        xe_addons.rotary_half_with_cache_inplaced(query_states, key_states, cos, sin)
    else:
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

    key_states, value_states = past_key_value.update(key_states, value_states,
                                                     self.layer_idx, None)

    attn_weights = None
    attn_output = scaled_dot_product_attention(
        query_states, key_states, value_states,
        attention_mask, q_len == key_states.size(2)
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


def qwen2_5_omni_thinker_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None
        else self.config.output_hidden_states
    )

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    invalidInputError((input_ids is None) ^ (inputs_embeds is None),
                      "You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # ipex-llm changes start: kv cache
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    use_cache = True if inputs_embeds.device.type == "xpu" else use_cache
    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and not torch.jit.is_tracing():
        if not isinstance(past_key_values, DynamicNormalCache):
            past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)
    # ipex-llm changes end

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    # ipex-llm changes start: rotary embedding
    if inputs_embeds.device.type == "xpu":
        cos, sin = position_embeddings
        mrope_section = self.config.rope_scaling["mrope_section"] * 2
        cos = torch.cat([m[i % 3] for i, m in
                        enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        sin = torch.cat([m[i % 3] for i, m in
                        enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        position_embeddings = cos.contiguous(), sin.contiguous()
    # ipex-llm changes end

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                     if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def qwen2_5_omni_vision_attention_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor = None
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
    k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
    v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
    # q, k, v: [seq_length, num_heads, head_dim]

    seq_lens = cu_seqlens.tolist()
    invalidInputError(seq_lens[0] == 0 and seq_lens[-1] == seq_length,
                      "unexpected input")

    head_dim = q.size(-1)
    if use_sdp_non_causal(head_dim, q.device, q.dtype):
        image_num = len(seq_lens) - 1
        image_size = seq_lens[1] - seq_lens[0]
        guessed_seq_lens = torch.arange(0, (image_num + 1) * image_size, image_size,
                                        dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        if (guessed_seq_lens == cu_seqlens).all():
            q = q.view(image_num, image_size, self.num_heads, head_dim).permute(0, 2, 1, 3)
            k = k.view(image_num, image_size, self.num_heads, head_dim).permute(0, 2, 1, 3)
            v = v.view(image_num, image_size, self.num_heads, head_dim).permute(0, 2, 1, 3)
            # q, k, v: [image_num, num_heads, image_size, head_dim]

            attn_output = scaled_dot_product_attention(
                q, k.contiguous(), v.contiguous(),
                None, False
            )
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
            attn_output = attn_output.view(seq_length, self.num_heads, head_dim)
            # attn_output: [seq_length, num_heads, head_dim]
        else:
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0).contiguous()
            v = v.transpose(0, 1).unsqueeze(0).contiguous()
            # q, k, v: [1, num_heads, seq_length, head_dim]

            attn_outputs = []
            for i in range(image_num):
                start_idx = seq_lens[i]
                end_idx = seq_lens[i + 1]
                tmp_q = q[:, :, start_idx:end_idx, :]
                tmp_k = k[:, :, start_idx:end_idx, :]
                tmp_v = v[:, :, start_idx:end_idx, :]
                attn_output = scaled_dot_product_attention(
                    tmp_q, tmp_k, tmp_v,
                    None, False
                )
                attn_output = attn_output.permute(0, 2, 1, 3)
                # attn_output: [1, seq_length, num_heads, head_dim]
                attn_outputs.append(attn_output)
            attn_output = torch.cat(attn_outputs, dim=1).squeeze(0)
            # attn_output: [seq_length, num_heads, head_dim]
    else:
        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(seq_lens)):
            attention_mask[..., seq_lens[i - 1]:seq_lens[i], seq_lens[i - 1]:seq_lens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        # q, k, v: [num_heads, seq_length, head_dim]

        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = attention_softmax(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        # attn_output: [seq_length, num_heads, head_dim]

    attn_output = attn_output.reshape(seq_length, -1)
    attn_output = self.proj(attn_output)
    return attn_output


def qwen2_5_omni_audio_attention_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[EncoderDecoderCache] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None
    seq_length, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states)
    query_states = query_states.reshape(seq_length, self.num_heads, -1)

    seq_lens = cu_seqlens.tolist()
    invalidInputError(seq_lens[0] == 0 and seq_lens[-1] == seq_length,
                      "unexpected input")

    if past_key_value is not None:
        is_updated = past_key_value.is_updated.get(self.layer_idx)
        if is_cross_attention:
            # after the first generated id,
            # we can subsequently re-use all key/value_states from cache
            past_key_value.is_updated[self.layer_idx] = True
            past_key_value = past_key_value.cross_attention_cache
        else:
            past_key_value = past_key_value.self_attention_cache

    # use key_value_states if cross attention
    current_states = key_value_states if key_value_states is not None else hidden_states
    if is_cross_attention and past_key_value and is_updated:
        # reuse k,v, cross_attentions
        key_states = past_key_value.key_cache[self.layer_idx]
        value_states = past_key_value.value_cache[self.layer_idx]
    else:
        key_states = self.k_proj(current_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(current_states).reshape(seq_length, self.num_heads, -1)
        if past_key_value is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

    if layer_head_mask is None and use_sdp_non_causal(query_states.size(-1),
                                                      query_states.device, query_states.dtype):
        kv_length = key_states.size(0)
        padding_kv_length = (kv_length + 128 - 1) // 128 * 128
        attention_mask = torch.full(
            [1, 1, seq_length, padding_kv_length], torch.finfo(query_states.dtype).min,
            device=query_states.device, dtype=query_states.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., seq_lens[i - 1]:seq_lens[i], seq_lens[i - 1]:seq_lens[i]] = 0

        q = query_states.transpose(0, 1).unsqueeze(0)
        k = key_states.transpose(0, 1).unsqueeze(0).contiguous()
        v = value_states.transpose(0, 1).unsqueeze(0).contiguous()
        # q, k, v: [1, num_heads, seq_length, head_dim]

        attn_weights = None
        attn_output = scaled_dot_product_attention(q, k, v, attention_mask, False)
        attn_output = attn_output.permute(0, 2, 1, 3).squeeze(0)
        # attn_output: [seq_length, num_heads, head_dim]
    else:
        attention_mask = torch.full(
            [1, seq_length, key_states.size(0)], torch.finfo(query_states.dtype).min,
            device=query_states.device, dtype=query_states.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., seq_lens[i - 1]:seq_lens[i], seq_lens[i - 1]:seq_lens[i]] = 0

        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)

        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = attention_softmax(attn_weights)

        if layer_head_mask is not None:
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_output = torch.matmul(attn_weights, value_states).transpose(0, 1)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state`s
    # because `attn_output` can be partitioned across GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(seq_length, self.embed_dim)
    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights, past_key_value


def dit_attention_forward(
    self,
    x,
    rope=None,
    mask=None,
) -> torch.Tensor:
    batch_size = x.shape[0]

    # `sample` projections.
    query = self.to_q(x)
    key = self.to_k(x)
    value = self.to_v(x)

    # attention
    inner_dim = key.shape[-1]
    head_dim = inner_dim // self.heads
    query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

    # apply rotary position embedding
    # Due to training process, only first head is applied with RoPE, will be fixed at next release
    cos, sin = rope
    query[:, :1], key[:, :1] = apply_rotary_pos_emb(query[:, :1], key[:, :1], cos, sin)

    if use_sdp_non_causal(head_dim, query.device, query.dtype):
        mask = torch.where(mask, 0, torch.finfo(query.dtype).min)
        x = scaled_dot_product_attention(query, key.contiguous(), value.contiguous(), mask, False)
        x = x.transpose(1, 2)
    else:
        attention_interface = ALL_ATTENTION_FUNCTIONS[self._attn_implementation]
        x, _ = attention_interface(self, query, key, value, attention_mask=mask, is_causal=False)

    # mask
    x = x.reshape(batch_size, -1, self.heads * head_dim)
    x = x.to(query.dtype)

    # linear proj
    x = self.to_out[0](x)
    # dropout
    x = self.to_out[1](x)

    return x


def _create_block_diff(self, x):
    batch, seq_len = x.shape[0], x.shape[1]
    block_indices = torch.arange(seq_len, device=x.device) // self.block_size

    block_i = block_indices.unsqueeze(1)  # [seq_length, 1]
    block_j = block_indices.unsqueeze(0)  # [1, seq_length]

    block_diff = block_j - block_i  # (n, n)
    return block_diff.unsqueeze(0).unsqueeze(0)
