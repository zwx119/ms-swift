# Copyright (c) Alibaba, Inc. and its affiliates.

"""Layer spec helpers for Swift DeltaNet experiments."""

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec

from .attention import DeltaNetSelfAttention


def get_deltanet_gpt_layer_spec(
    *,
    use_transformer_engine: bool,
    normalization: str,
    num_experts=None,
    moe_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
    multi_latent_attention: bool = False,
    moe_use_legacy_grouped_gemm: bool = False,
) -> ModuleSpec:
    """Return the regular GPT layer spec with self-attention swapped to DeltaNet."""

    if use_transformer_engine:
        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            qk_layernorm=qk_layernorm,
            multi_latent_attention=multi_latent_attention,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        )
    else:
        layer_spec = get_gpt_layer_local_spec(
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            qk_layernorm=qk_layernorm,
            multi_latent_attention=multi_latent_attention,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
            normalization=normalization,
        )

    softmax_attn = layer_spec.submodules.self_attention
    layer_spec.submodules.self_attention = ModuleSpec(
        module=DeltaNetSelfAttention,
        params={'attn_mask_type': AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv=softmax_attn.submodules.linear_qkv,
            core_attention=None,
            linear_proj=softmax_attn.submodules.linear_proj,
            q_layernorm=None,
            k_layernorm=None,
        ),
    )
    return layer_spec
