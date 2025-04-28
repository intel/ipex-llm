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

import torch
from typing import Optional, List, Tuple
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3Attention

from ipex_llm.transformers.kv import DynamicNormalCache
from ipex_llm.transformers.models.common import merge_qkv_base
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.models.utils import make_cache_contiguous_inplaced


def merge_qkv(module: torch.nn.Module):
    merge_qkv_base(module, Qwen3Attention)


def qwen3_model_forward(
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

    return Qwen3Model.forward(
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


def qwen3_attention_forward(
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
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, cache_kwargs)
    attn_weights = None
    attn_output = scaled_dot_product_attention(
        query_states, key_states, value_states,
        attention_mask, q_len == key_states.size(2), self.scaling
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
