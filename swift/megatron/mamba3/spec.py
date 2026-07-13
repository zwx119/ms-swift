# Copyright (c) Alibaba, Inc. and its affiliates.

"""Layer spec helpers for Swift Mamba3 experiments."""

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    LayerNormImpl,
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

try:
    from megatron.core.extensions.transformer_engine import TENorm
except ImportError:  # pragma: no cover - only used when TE is unavailable.
    TENorm = None

from .attention import Mamba3Attention
from ..deltanet.attention import Seq1F1BHybridSelfAttention


def _base_gpt_layer_spec(
    *,
    use_transformer_engine: bool,
    normalization: str,
    num_experts=None,
    moe_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
    multi_latent_attention: bool = False,
    moe_use_legacy_grouped_gemm: bool = False,
) -> ModuleSpec:
    if use_transformer_engine:
        return get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            qk_layernorm=qk_layernorm,
            multi_latent_attention=multi_latent_attention,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        )
    return get_gpt_layer_local_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        qk_layernorm=qk_layernorm,
        multi_latent_attention=multi_latent_attention,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        normalization=normalization,
    )


def _parse_layer_selection(spec: str) -> set[int]:
    layers: set[int] = set()
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if '-' in item:
            start_s, end_s = item.split('-', 1)
            start = int(start_s)
            end = int(end_s)
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(f'invalid hybrid attention layer range: {item}')
            layers.update(range(start, end + 1))
        else:
            layer = int(item)
            if layer <= 0:
                raise ValueError('hybrid attention layers are 1-indexed')
            layers.add(layer)
    return layers


def _softmax_layers(num_layers: int, explicit_layers: str, period: int, offset: int) -> set[int]:
    layers = _parse_layer_selection(explicit_layers)
    if period > 0:
        for layer in range(1, num_layers + 1):
            if (layer - offset) % period == 0:
                layers.add(layer)
    return {layer for layer in layers if 1 <= layer <= num_layers}


def _with_mamba3_attention(layer_spec: ModuleSpec, *, use_transformer_engine: bool) -> ModuleSpec:
    if use_transformer_engine:
        if TENorm is None:
            raise RuntimeError('TransformerEngine Mamba3 spec requires TENorm.')
        # TE GPT attention fuses input layernorm into TELayerNormColumnParallelLinear.
        # Mamba3Attention owns its official Mamba3 projections and does not consume
        # that fused qkv module, so restore the skipped attention input norm explicitly.
        layer_spec.submodules.input_layernorm = TENorm
    softmax_attn = layer_spec.submodules.self_attention
    layer_spec.submodules.self_attention = ModuleSpec(
        module=Mamba3Attention,
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


def _with_seq1f1b_hybrid_softmax_attention(layer_spec: ModuleSpec) -> ModuleSpec:
    softmax_attn = layer_spec.submodules.self_attention
    layer_spec.submodules.self_attention = ModuleSpec(
        module=Seq1F1BHybridSelfAttention,
        params={'attn_mask_type': AttnMaskType.causal},
        submodules=softmax_attn.submodules,
    )
    return layer_spec


def get_mamba3_gpt_layer_spec(
    *,
    use_transformer_engine: bool,
    normalization: str,
    num_experts=None,
    moe_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
    multi_latent_attention: bool = False,
    moe_use_legacy_grouped_gemm: bool = False,
    config=None,
    num_layers: int = 0,
    hybrid_attention_layers: str = '',
    hybrid_attention_period: int = 0,
    hybrid_attention_offset: int = 0,
    **_,
) -> ModuleSpec:
    softmax_layers = _softmax_layers(
        num_layers,
        hybrid_attention_layers,
        hybrid_attention_period,
        hybrid_attention_offset,
    )
    if not softmax_layers:
        return _with_mamba3_attention(
            _base_gpt_layer_spec(
                use_transformer_engine=use_transformer_engine,
                normalization=normalization,
                num_experts=num_experts,
                moe_grouped_gemm=moe_grouped_gemm,
                qk_layernorm=qk_layernorm,
                multi_latent_attention=multi_latent_attention,
                moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
            ),
            use_transformer_engine=use_transformer_engine,
        )

    layer_specs = []
    for layer_number in range(1, num_layers + 1):
        layer_spec = _base_gpt_layer_spec(
            use_transformer_engine=use_transformer_engine,
            normalization=normalization,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            qk_layernorm=qk_layernorm,
            multi_latent_attention=multi_latent_attention,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        )
        if layer_number in softmax_layers:
            layer_spec = _with_seq1f1b_hybrid_softmax_attention(layer_spec)
        else:
            layer_spec = _with_mamba3_attention(layer_spec, use_transformer_engine=use_transformer_engine)
        layer_specs.append(layer_spec)

    if config is not None:
        offset = get_transformer_layer_offset(config)
        num_layers_to_build = get_num_layers_to_build(config)
        layer_specs = layer_specs[offset:offset + num_layers_to_build]

    return TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=LayerNormImpl)
