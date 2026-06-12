# Copyright (c) Alibaba, Inc. and its affiliates.

"""MCore-compatible DeltaNet attention with Seq1F1B state relay."""

import inspect
from typing import Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide
from megatron.training import get_args

from .context import get_seq_split_context

try:
    from einops import rearrange
except ImportError:  # pragma: no cover - validated on GPU image
    rearrange = None

try:
    from fla.modules import FusedRMSNormGated, RMSNorm as FLARMSNorm, ShortConvolution
    from fla.modules.conv.triton.ops import causal_conv1d_bwd, causal_conv1d_fwd
    from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
    from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
    from fla.ops.delta_rule.chunk import chunk_delta_rule_bwd, chunk_delta_rule_fwd

    HAS_FLA = True
except ImportError:  # pragma: no cover - validated on GPU image
    HAS_FLA = False
    warnings.warn(
        'flash-linear-attention (fla) is not installed. '
        'DeltaNet attention can only be constructed after installing fla.'
    )


def _supports_kwarg(fn, name: str) -> bool:
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)


_CHUNK_DELTA_RULE_FWD_SUPPORTS_HO_PIPELINE = (
    HAS_FLA and _supports_kwarg(chunk_delta_rule_fwd, 'use_ho_pipeline')
)
_CHUNK_DELTA_RULE_SUPPORTS_HO_PIPELINE = (
    HAS_FLA and _supports_kwarg(chunk_delta_rule, 'use_ho_pipeline')
)


class ShortConvChunkFunc(torch.autograd.Function):
    """FLA causal conv1d with explicit cross-chunk gradient relay."""

    @staticmethod
    def forward(ctx, x, weight, bias, cache_dict, grad_dict, name, activation):
        weight_dw = rearrange(weight, 'd 1 w -> d w')
        initial_state = cache_dict.get(name, None)
        had_state = initial_state is not None
        y, final_state = causal_conv1d_fwd(
            x=x,
            weight=weight_dw,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            output_final_state=True,
            activation=activation,
        )
        # The cache is a Python-side boundary: final_state is not returned as a
        # tensor argument to the next chunk's autograd Function. Backward state
        # relay is handled explicitly through grad_dict.
        cache_dict[name] = final_state

        ctx.save_for_backward(x, weight_dw, bias, initial_state)
        ctx.activation = activation
        ctx.grad_dict = grad_dict
        ctx.name = name
        ctx.had_state = had_state
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight_dw, bias, initial_state = ctx.saved_tensors
        dht = ctx.grad_dict.pop(ctx.name, None)
        dx, dw, db, _, dh0 = causal_conv1d_bwd(
            x=x,
            dy=dy,
            dht=dht,
            weight=weight_dw,
            bias=bias,
            initial_state=initial_state,
            activation=ctx.activation,
        )
        if ctx.had_state and dh0 is not None:
            ctx.grad_dict[ctx.name] = dh0
        return dx, rearrange(dw, 'd w -> d 1 w'), db, None, None, None, None


class DeltaNetChunkFunc(torch.autograd.Function):
    """FLA chunk delta rule with explicit recurrent-state gradient relay.

    The recurrent state is intentionally kept in ``state_cache`` instead of
    being passed as a tensor argument. That prevents autograd from wiring chunk
    N to chunk N-1 while still allowing backward to hand ``dh0`` to the previous
    chunk through the same cache.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        beta,
        scale,
        state_cache,
        use_qk_l2norm_in_kernel,
        use_ho_pipeline,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        initial_state = state_cache.get('recurrent_state', None)
        fwd_kwargs = {
            'q': q,
            'k': k,
            'v': v,
            'beta': beta,
            'scale': scale,
            'initial_state': initial_state,
            'output_final_state': True,
            'cu_seqlens': None,
            'chunk_indices': None,
        }
        if _CHUNK_DELTA_RULE_FWD_SUPPORTS_HO_PIPELINE:
            fwd_kwargs['use_ho_pipeline'] = use_ho_pipeline
        o, A, final_state = chunk_delta_rule_fwd(**fwd_kwargs)
        # Match the stateflow pattern: recurrent state is relayed through a
        # Python dict, outside autograd's tensor-argument graph. Its gradient is
        # passed manually via state_cache['d_state'] in backward().
        state_cache['recurrent_state'] = final_state

        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, A)
        ctx.initial_state = initial_state
        ctx.scale = scale
        ctx.state_cache = state_cache
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype)

    @staticmethod
    def backward(ctx, do):
        dht = ctx.state_cache.pop('d_state', None)
        q, q_rstd, k, k_rstd, v, beta, A = ctx.saved_tensors

        dq, dk, dv, db, dh0 = chunk_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=ctx.initial_state,
            do=do,
            dht=dht,
            cu_seqlens=None,
            chunk_indices=None,
        )
        if dh0 is not None:
            ctx.state_cache['d_state'] = dh0

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), None, None, None, None


class DeltaNetSelfAttention(MegatronModule):
    """Drop-in self-attention replacement for MCore TransformerLayer."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: str = None,  # kept for MCore build_module compatibility
    ):
        super().__init__(config=config)
        if not HAS_FLA or rearrange is None:
            raise ImportError('DeltaNet requires flash-linear-attention and einops on the GPU image.')

        args = get_args()
        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = config.sequence_parallel

        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_attention_heads_per_partition = divide(config.num_attention_heads, world_size)
        self.head_dim = divide(config.hidden_size, config.num_attention_heads)
        self.local_hidden_size = self.num_attention_heads_per_partition * self.head_dim

        self.use_short_conv = getattr(args, 'deltanet_use_short_conv', True)
        self.conv_size = getattr(args, 'deltanet_conv_size', 4)
        self.use_beta = getattr(args, 'deltanet_use_beta', True)
        self.use_gate = getattr(args, 'deltanet_use_output_gate', True)
        self.qk_activation = getattr(args, 'deltanet_qk_activation', 'silu')
        self.qk_norm = getattr(args, 'deltanet_qk_norm', 'l2')
        self.deltanet_mode = getattr(args, 'deltanet_mode', 'chunk')
        self.use_ho_pipeline = getattr(args, 'deltanet_fused_h_o_pipeline', False)

        if self.use_gate:
            qkvg_kwargs = dict(
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='qkvg',
            )
            try:
                self.linear_qkvg = build_module(
                    submodules.linear_qkv,
                    self.hidden_size,
                    4 * self.hidden_size,
                    stride=4,
                    **qkvg_kwargs,
                )
            except TypeError:
                if world_size > 1:
                    raise TypeError(
                        'DeltaNet fused qkvg projection needs a strided column-parallel linear for TP>1. '
                        'Use TP=1 or transformer_impl=local.'
                    )
                self.linear_qkvg = build_module(
                    submodules.linear_qkv,
                    self.hidden_size,
                    4 * self.hidden_size,
                    **qkvg_kwargs,
                )
        else:
            self.linear_q = build_module(
                submodules.linear_qkv,
                self.hidden_size,
                self.hidden_size,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q',
            )
            self.linear_k = build_module(
                submodules.linear_qkv,
                self.hidden_size,
                self.hidden_size,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='k',
            )
            self.linear_v = build_module(
                submodules.linear_qkv,
                self.hidden_size,
                self.hidden_size,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='v',
            )

        if self.use_beta:
            self.b_proj = nn.Linear(self.hidden_size, self.num_attention_heads_per_partition, bias=False)

        if self.use_short_conv:
            qk_act = 'silu' if self.qk_activation == 'silu' else None
            self.q_conv1d = ShortConvolution(
                hidden_size=self.local_hidden_size,
                kernel_size=self.conv_size,
                bias=False,
                activation=qk_act,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.local_hidden_size,
                kernel_size=self.conv_size,
                bias=False,
                activation=qk_act,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.local_hidden_size,
                kernel_size=self.conv_size,
                bias=False,
                activation='silu',
            )

        if self.use_gate:
            self.o_norm = FusedRMSNormGated(self.head_dim, eps=config.layernorm_epsilon)
        else:
            self.o_norm = FLARMSNorm(self.head_dim, eps=config.layernorm_epsilon, dtype=torch.float32)

        self.linear_proj = build_module(
            submodules.linear_proj,
            self.hidden_size,
            self.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )

        self.state_cache = {}
        self.conv_cache_dict = {}
        self.conv_grad_dict = {}

    def _clear_states(self) -> None:
        self.state_cache = {}
        self.conv_cache_dict = {}
        self.conv_grad_dict = {}

    def _project_qkvg(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.use_gate:
            qkvg, _ = self.linear_qkvg(hidden_states)
            q, k, v, g = torch.chunk(qkvg, 4, dim=-1)
        else:
            q, _ = self.linear_q(hidden_states)
            k, _ = self.linear_k(hidden_states)
            v, _ = self.linear_v(hidden_states)
            g = None
        return (
            q.transpose(0, 1).contiguous(),
            k.transpose(0, 1).contiguous(),
            v.transpose(0, 1).contiguous(),
            None if g is None else g.transpose(0, 1).contiguous(),
        )

    def _hidden_states_for_beta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.sequence_parallel and parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=True
            )
        return hidden_states.transpose(0, 1).contiguous()

    def _apply_short_conv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output_final_state: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_short_conv:
            if self.qk_activation == 'silu':
                q, k = F.silu(q), F.silu(k)
            elif self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q = F.elu(q, 1.0, False) + 1.0
                k = F.elu(k, 1.0, False) + 1.0
            return q, k, F.silu(v)

        qk_act = 'silu' if self.qk_activation == 'silu' else None
        if output_final_state:
            q = ShortConvChunkFunc.apply(
                q, self.q_conv1d.weight, self.q_conv1d.bias, self.conv_cache_dict, self.conv_grad_dict, 'q', qk_act
            )
            k = ShortConvChunkFunc.apply(
                k, self.k_conv1d.weight, self.k_conv1d.bias, self.conv_cache_dict, self.conv_grad_dict, 'k', qk_act
            )
            v = ShortConvChunkFunc.apply(
                v, self.v_conv1d.weight, self.v_conv1d.bias, self.conv_cache_dict, self.conv_grad_dict, 'v', 'silu'
            )
        else:
            q, _ = self.q_conv1d(q, cache=None, output_final_state=False)
            k, _ = self.k_conv1d(k, cache=None, output_final_state=False)
            v, _ = self.v_conv1d(v, cache=None, output_final_state=False)

        if self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q = F.elu(q, 1.0, False) + 1.0
                k = F.elu(k, 1.0, False) + 1.0
        return q, k, v

    def _delta_rule(self, q, k, v, beta, output_final_state: bool) -> torch.Tensor:
        seq_len = q.shape[1]
        mode = 'fused_recurrent' if seq_len <= 64 else self.deltanet_mode
        initial_state = self.state_cache.get('recurrent_state', None)

        if output_final_state and mode == 'chunk':
            orig_dtype = q.dtype
            if orig_dtype == torch.float32:
                q, k, v, beta = q.bfloat16(), k.bfloat16(), v.bfloat16(), beta.bfloat16()
            out = DeltaNetChunkFunc.apply(
                q,
                k,
                v,
                beta,
                self.head_dim**-0.5,
                self.state_cache,
                self.qk_norm == 'l2',
                self.use_ho_pipeline,
            )
            return out.to(orig_dtype)

        if mode == 'fused_recurrent':
            out, recurrent_state = fused_recurrent_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=self.qk_norm == 'l2',
            )
        elif mode == 'chunk':
            orig_dtype = q.dtype
            if orig_dtype == torch.float32:
                q, k, v, beta = q.bfloat16(), k.bfloat16(), v.bfloat16(), beta.bfloat16()
            chunk_kwargs = {
                'q': q,
                'k': k,
                'v': v,
                'beta': beta,
                'scale': self.head_dim**-0.5,
                'initial_state': initial_state,
                'output_final_state': output_final_state,
                'use_qk_l2norm_in_kernel': self.qk_norm == 'l2',
            }
            if _CHUNK_DELTA_RULE_SUPPORTS_HO_PIPELINE:
                chunk_kwargs['use_ho_pipeline'] = self.use_ho_pipeline
            out, recurrent_state = chunk_delta_rule(**chunk_kwargs)
            out = out.to(orig_dtype)
        else:
            raise NotImplementedError(f'DeltaNet mode `{mode}` is not supported.')

        if output_final_state and recurrent_state is not None:
            self.state_cache['recurrent_state'] = recurrent_state
        return out

    def _forward_delta(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        split_context = get_seq_split_context()
        if split_context.micro_sp_idx == 0:
            self._clear_states()

        output_final_state = split_context.pipe_sp_splits > 1
        q, k, v, g = self._project_qkvg(hidden_states)
        hidden_states_bsh = self._hidden_states_for_beta(hidden_states)

        q, k, v = self._apply_short_conv(q, k, v, output_final_state=output_final_state)
        q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)
        k = rearrange(k, 'b s (h d) -> b s h d', d=self.head_dim)
        v = rearrange(v, 'b s (h d) -> b s h d', d=self.head_dim)

        if self.use_beta:
            beta = self.b_proj(hidden_states_bsh).sigmoid()
        else:
            beta = torch.ones(
                hidden_states_bsh.shape[0],
                hidden_states_bsh.shape[1],
                self.num_attention_heads_per_partition,
                dtype=hidden_states_bsh.dtype,
                device=hidden_states_bsh.device,
            )

        out = self._delta_rule(q, k, v, beta, output_final_state=output_final_state)
        if self.use_gate:
            g = rearrange(g, 'b s (h d) -> b s h d', d=self.head_dim)
            out = self.o_norm(out, g)
        else:
            out = self.o_norm(out)

        out = rearrange(out, 'b s h d -> s b (h d)').contiguous()
        return self.linear_proj(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del attention_mask, key_value_states, inference_context, rotary_pos_emb
        del rotary_pos_cos, rotary_pos_sin, attention_bias, sequence_len_offset, inference_params
        if packed_seq_params is not None:
            args = get_args()
            if not getattr(args, 'deltanet_allow_packed_seq', False):
                raise RuntimeError(
                    'DeltaNet recurrent state is not compatible with packed sequence resets yet. '
                    'Run with --packing false, or set --deltanet-allow-packed-seq true only for throughput-only tests.'
                )
        return self._forward_delta(hidden_states)
