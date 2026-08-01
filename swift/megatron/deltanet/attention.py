# Copyright (c) Alibaba, Inc. and its affiliates.

"""MCore-compatible DeltaNet attention with Seq1F1B state relay."""

import math
import os
from typing import Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide
from megatron.model.seq1f1b_kv_relay import (
    accumulate_prefix_gradients,
    drop_active_kv,
    is_recompute_cache_sealed,
    record_prefix_offset,
    restore_recompute_prefix,
    seal_recompute_kv,
)
from megatron.training import get_args

from .context import get_seq_split_context

try:
    from einops import rearrange
except ImportError:  # pragma: no cover - validated on GPU image
    rearrange = None

try:
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward, _flash_attn_varlen_forward

    HAS_FLASH_ATTN = True
except ImportError:  # pragma: no cover - validated on GPU image
    _flash_attn_varlen_backward = None
    _flash_attn_varlen_forward = None
    HAS_FLASH_ATTN = False

try:
    from fla.modules import FusedRMSNormGated, RMSNorm as FLARMSNorm, ShortConvolution
    from fla.modules.conv.triton.ops import causal_conv1d_bwd, causal_conv1d_fwd
    from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd

    HAS_FLA = True
except ImportError:  # pragma: no cover - validated on GPU image
    HAS_FLA = False
    warnings.warn(
        'flash-linear-attention (fla) is not installed. '
        'DeltaNet attention can only be constructed after installing fla.'
    )


class ShortConvChunkFunc(torch.autograd.Function):
    """FLA causal conv1d with explicit cross-chunk gradient relay."""

    @staticmethod
    def forward(ctx, x, weight, bias, cache_dict, grad_dict, name, activation):
        weight_dw = rearrange(weight, 'd 1 w -> d w')
        replay = cache_dict.get('_recompute_replay', False)
        if replay:
            initial_state = cache_dict.get('_recompute_initial_states', {}).get(name, None)
        else:
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
        # relay is handled explicitly through grad_dict. During activation
        # recompute replay, do not overwrite the real forward's boundary state.
        if not replay:
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
    """FLA chunk gated delta rule with explicit recurrent-state gradient relay.

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
        recurrent_gate,
        beta,
        scale,
        state_cache,
        use_qk_l2norm_in_kernel,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        replay = state_cache.get('_recompute_replay', False)
        if replay:
            initial_state = state_cache.get('_recompute_initial_state', None)
        else:
            initial_state = state_cache.get('recurrent_state', None)
        fwd_out = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=recurrent_gate,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=None,
            chunk_indices=None,
        )
        if len(fwd_out) == 5:
            recurrent_gate, o, A, final_state, initial_state = fwd_out
        elif len(fwd_out) == 6:
            recurrent_gate, o, A, final_state, initial_state, _ = fwd_out
        else:
            raise RuntimeError(f'Unexpected chunk_gated_delta_rule_fwd return size: {len(fwd_out)}')
        # Match the stateflow pattern: recurrent state is relayed through a
        # Python dict, outside autograd's tensor-argument graph. Its gradient is
        # passed manually via state_cache['d_state'] in backward(). During
        # activation recompute replay, do not overwrite the real forward state.
        if not replay:
            state_cache['recurrent_state'] = final_state

        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, recurrent_gate, beta, A)
        ctx.initial_state = initial_state
        ctx.scale = scale
        ctx.state_cache = state_cache
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype)

    @staticmethod
    def backward(ctx, do):
        dht = ctx.state_cache.pop('d_state', None)
        q, q_rstd, k, k_rstd, v, recurrent_gate, beta, A = ctx.saved_tensors

        bwd_out = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=recurrent_gate,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=ctx.initial_state,
            do=do,
            dht=dht,
            cu_seqlens=None,
            chunk_indices=None,
        )
        if len(bwd_out) < 6:
            raise RuntimeError(f'Unexpected chunk_gated_delta_rule_bwd return size: {len(bwd_out)}')
        dq, dk, dv, db, dg, dh0 = bwd_out[:6]
        if dh0 is not None:
            ctx.state_cache['d_state'] = dh0

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dg.to(recurrent_gate.dtype),
            db.to(beta.dtype),
            None,
            None,
            None,
        )


class DeltaNetSelfAttention(MegatronModule):
    seq1f1b_accepts_microbatch_cache_key = True

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

        self.a_proj = nn.Linear(self.hidden_size, self.num_attention_heads_per_partition, bias=False)
        A = torch.empty(self.num_attention_heads_per_partition, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_attention_heads_per_partition) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

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
        self._recompute_state_snapshots = {}
        self._recompute_conv_snapshots = {}

    @staticmethod
    def _clone_cache_tensor(tensor):
        return None if tensor is None else tensor.detach().clone()

    def _clear_states(self, clear_recompute_snapshots: bool = True) -> None:
        self.state_cache = {}
        self.conv_cache_dict = {}
        self.conv_grad_dict = {}
        if clear_recompute_snapshots:
            self._recompute_state_snapshots = {}
            self._recompute_conv_snapshots = {}

    def _full_recompute_seq1f1b_enabled(self, args, micro_sp_idx) -> bool:
        return (
            self.training
            and micro_sp_idx is not None
            and getattr(args, 'pipe_sp_splits', 1) > 1
            and getattr(args, 'recompute_granularity', None) == 'full'
            and not getattr(args, 'deltanet_rnn_sp1_baseline', False)
        )

    def _recompute_debug(self, phase, micro_sp_idx, state_stack=None, conv_stack=None) -> None:
        if os.environ.get('DELTANET_RECOMPUTE_DEBUG', '0') == '0':
            return
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        except Exception:
            rank = -1
        layer = getattr(self, 'layer_number', None)
        if micro_sp_idx != 7 and phase != 'missing':
            return
        state_keys = sorted([k for k in self._recompute_state_snapshots.keys()])
        state_counts = {k: len(v) for k, v in self._recompute_state_snapshots.items()}
        conv_counts = {k: len(v) for k, v in self._recompute_conv_snapshots.items()}
        state_len = None if state_stack is None else len(state_stack)
        conv_len = None if conv_stack is None else len(conv_stack)
        print(
            f'[deltanet_recompute_debug] phase={phase} rank={rank} layer={layer} '
            f'chunk={micro_sp_idx} grad={torch.is_grad_enabled()} '
            f'state_len={state_len} conv_len={conv_len} keys={state_keys} '
            f'state_counts={state_counts} conv_counts={conv_counts}',
            flush=True,
        )

    def _begin_recompute_safe_state(self, args, micro_sp_idx: int) -> bool:
        if not self._full_recompute_seq1f1b_enabled(args, micro_sp_idx):
            return False

        if not torch.is_grad_enabled():
            self._recompute_state_snapshots.setdefault(micro_sp_idx, []).append(
                self._clone_cache_tensor(self.state_cache.get('recurrent_state', None))
            )
            self._recompute_conv_snapshots.setdefault(micro_sp_idx, []).append({
                name: self._clone_cache_tensor(self.conv_cache_dict.get(name, None))
                for name in ('q', 'k', 'v')
            })
            self._recompute_debug('record', micro_sp_idx)
            return False

        state_stack = self._recompute_state_snapshots.get(micro_sp_idx)
        conv_stack = self._recompute_conv_snapshots.get(micro_sp_idx)
        self._recompute_debug('replay_before', micro_sp_idx, state_stack, conv_stack)
        if state_stack is None or len(state_stack) == 0:
            self._recompute_debug('missing', micro_sp_idx, state_stack, conv_stack)
            raise RuntimeError(f'Missing DeltaNet recompute recurrent-state snapshot for chunk {micro_sp_idx}.')
        if conv_stack is None or len(conv_stack) == 0:
            self._recompute_debug('missing', micro_sp_idx, state_stack, conv_stack)
            raise RuntimeError(f'Missing DeltaNet recompute conv-state snapshot for chunk {micro_sp_idx}.')
        self.state_cache['_recompute_replay'] = True
        self.state_cache['_recompute_initial_state'] = state_stack.pop()
        self.conv_cache_dict['_recompute_replay'] = True
        self.conv_cache_dict['_recompute_initial_states'] = conv_stack.pop()
        self._recompute_debug('replay_after', micro_sp_idx, state_stack, conv_stack)
        return True

    def _end_recompute_safe_state(self, recompute_replay: bool) -> None:
        if not recompute_replay:
            return
        self.state_cache.pop('_recompute_replay', None)
        self.state_cache.pop('_recompute_initial_state', None)
        self.conv_cache_dict.pop('_recompute_replay', None)
        self.conv_cache_dict.pop('_recompute_initial_states', None)

    def _project_qkvg(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.use_gate:
            qkvg, _ = self.linear_qkvg(hidden_states)
            q, k, v, output_gate = torch.chunk(qkvg, 4, dim=-1)
        else:
            q, _ = self.linear_q(hidden_states)
            k, _ = self.linear_k(hidden_states)
            v, _ = self.linear_v(hidden_states)
            output_gate = None
        return (
            q.transpose(0, 1).contiguous(),
            k.transpose(0, 1).contiguous(),
            v.transpose(0, 1).contiguous(),
            None if output_gate is None else output_gate.transpose(0, 1).contiguous(),
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

    def _recurrent_gate(self, hidden_states_bsh: torch.Tensor) -> torch.Tensor:
        return -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states_bsh).float() + self.dt_bias)

    def _delta_rule(self, q, k, v, recurrent_gate, beta, output_final_state: bool) -> torch.Tensor:
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
                recurrent_gate,
                beta,
                self.head_dim**-0.5,
                self.state_cache,
                self.qk_norm == 'l2',
            )
            return out.to(orig_dtype)

        if mode == 'fused_recurrent':
            out, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=recurrent_gate,
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
                'g': recurrent_gate,
                'beta': beta,
                'scale': self.head_dim**-0.5,
                'initial_state': initial_state,
                'output_final_state': output_final_state,
                'use_qk_l2norm_in_kernel': self.qk_norm == 'l2',
            }
            out, recurrent_state = chunk_gated_delta_rule(**chunk_kwargs)
            out = out.to(orig_dtype)
        else:
            raise NotImplementedError(f'DeltaNet mode `{mode}` is not supported.')

        if output_final_state and recurrent_state is not None:
            self.state_cache['recurrent_state'] = recurrent_state
        return out

    def _forward_delta(
        self,
        hidden_states: torch.Tensor,
        seq1f1b_micro_sp_idx=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        split_context = get_seq_split_context()
        output_final_state = split_context.pipe_sp_splits > 1
        args = get_args()
        micro_sp_idx = split_context.micro_sp_idx
        if seq1f1b_micro_sp_idx is not None:
            if torch.is_tensor(seq1f1b_micro_sp_idx):
                seq1f1b_micro_sp_idx = seq1f1b_micro_sp_idx.item()
            seq1f1b_micro_sp_idx = int(seq1f1b_micro_sp_idx)
            if seq1f1b_micro_sp_idx >= 0:
                micro_sp_idx = seq1f1b_micro_sp_idx
        if torch.is_tensor(micro_sp_idx):
            micro_sp_idx = micro_sp_idx.item()
        full_recompute_seq1f1b = self._full_recompute_seq1f1b_enabled(args, micro_sp_idx)
        recompute_replay = False
        if micro_sp_idx == 0 and not (full_recompute_seq1f1b and torch.is_grad_enabled()):
            self._clear_states(clear_recompute_snapshots=not full_recompute_seq1f1b)
        recompute_replay = self._begin_recompute_safe_state(args, micro_sp_idx)

        try:
            q, k, v, output_gate = self._project_qkvg(hidden_states)
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

            recurrent_gate = self._recurrent_gate(hidden_states_bsh)
            out = self._delta_rule(q, k, v, recurrent_gate, beta, output_final_state=output_final_state)
            if self.use_gate:
                output_gate = rearrange(output_gate, 'b s (h d) -> b s h d', d=self.head_dim)
                out = self.o_norm(out, output_gate)
            else:
                out = self.o_norm(out)

            out = rearrange(out, 'b s h d -> s b (h d)').contiguous()
            return self.linear_proj(out)
        finally:
            self._end_recompute_safe_state(recompute_replay)

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
        seq1f1b_microbatch_cache_key=None,
        seq1f1b_micro_sp_idx=None,
        seq1f1b_start=None,
        seq1f1b_end=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del attention_mask, key_value_states, inference_context, rotary_pos_emb
        del rotary_pos_cos, rotary_pos_sin, attention_bias, sequence_len_offset, inference_params
        del seq1f1b_microbatch_cache_key, seq1f1b_start, seq1f1b_end
        if packed_seq_params is not None:
            args = get_args()
            if not getattr(args, 'deltanet_allow_packed_seq', False):
                raise RuntimeError(
                    'DeltaNet recurrent state is not compatible with packed sequence resets yet. '
                    'Run with --packing false, or set --deltanet-allow-packed-seq true only for throughput-only tests.'
                )
        return self._forward_delta(hidden_states, seq1f1b_micro_sp_idx=seq1f1b_micro_sp_idx)


class Seq1F1BFlashAttnVarlenFunc(torch.autograd.Function):
    """FlashAttention full-prefix causal attention with chunkwise KV gradient relay."""

    @staticmethod
    def _flatten_batch_seq(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.size(0) * x.size(1), *x.shape[2:]).contiguous()

    @staticmethod
    def forward(ctx, query, key, value, kv_cache, dropout_p, causal, softmax_scale):
        if softmax_scale is None:
            softmax_scale = query.shape[-1] ** -0.5

        batch_size = query.size(0)
        seqlen_q = query.size(1)
        cached_key = kv_cache.get('key')
        cached_value = kv_cache.get('value')
        if cached_key is None:
            offset = 0
            full_key = key
            full_value = value
        else:
            offset = cached_key.size(1)
            full_key = torch.cat((cached_key, key), dim=1).contiguous()
            full_value = torch.cat((cached_value, value), dim=1).contiguous()

        kv_cache['key'] = full_key
        kv_cache['value'] = full_value

        seqlen_k = full_key.size(1)
        query_flat = Seq1F1BFlashAttnVarlenFunc._flatten_batch_seq(query)
        key_flat = Seq1F1BFlashAttnVarlenFunc._flatten_batch_seq(full_key)
        value_flat = Seq1F1BFlashAttnVarlenFunc._flatten_batch_seq(full_value)
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * seqlen_q,
            step=seqlen_q,
            dtype=torch.int32,
            device=query.device,
        )
        cu_seqlens_k = torch.arange(
            0,
            (batch_size + 1) * seqlen_k,
            step=seqlen_k,
            dtype=torch.int32,
            device=query.device,
        )

        out, softmax_lse, _, rng_state = _flash_attn_varlen_forward(
            query_flat,
            key_flat,
            value_flat,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )

        ctx.save_for_backward(query_flat, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.kv_cache = kv_cache
        ctx.batch_size = batch_size
        ctx.seqlen_q = seqlen_q
        ctx.seqlen_k = seqlen_k
        ctx.offset = offset
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out.reshape(batch_size, seqlen_q, *out.shape[1:])

    @staticmethod
    def backward(ctx, dout):
        query_flat, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        full_key = ctx.kv_cache['key'][:, :ctx.seqlen_k].contiguous()
        full_value = ctx.kv_cache['value'][:, :ctx.seqlen_k].contiguous()

        recompute_relay = is_recompute_cache_sealed(ctx.kv_cache)
        if recompute_relay:
            drop_active_kv(ctx.kv_cache, key_name='key', value_name='value')
        elif ctx.offset > 0:
            ctx.kv_cache['key'] = full_key[:, :ctx.offset].contiguous()
            ctx.kv_cache['value'] = full_value[:, :ctx.offset].contiguous()
        else:
            drop_active_kv(ctx.kv_cache, key_name='key', value_name='value')

        key_flat = Seq1F1BFlashAttnVarlenFunc._flatten_batch_seq(full_key)
        value_flat = Seq1F1BFlashAttnVarlenFunc._flatten_batch_seq(full_value)
        dout_flat = Seq1F1BFlashAttnVarlenFunc._flatten_batch_seq(dout)
        dq_flat = torch.empty_like(query_flat)
        dk_flat = torch.empty_like(key_flat)
        dv_flat = torch.empty_like(value_flat)

        _flash_attn_varlen_backward(
            dout_flat,
            query_flat,
            key_flat,
            value_flat,
            out,
            softmax_lse,
            dq_flat,
            dk_flat,
            dv_flat,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.seqlen_q,
            ctx.seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            rng_state=rng_state,
        )

        q_head_shape = query_flat.shape[1:]
        k_head_shape = key_flat.shape[1:]
        v_head_shape = value_flat.shape[1:]
        dq = dq_flat[..., :dout_flat.shape[-1]].reshape(ctx.batch_size, ctx.seqlen_q, *q_head_shape).contiguous()
        dk = dk_flat[..., :dout_flat.shape[-1]].reshape(ctx.batch_size, ctx.seqlen_k, *k_head_shape).contiguous()
        dv = dv_flat[..., :dout_flat.shape[-1]].reshape(ctx.batch_size, ctx.seqlen_k, *v_head_shape).contiguous()

        if recompute_relay:
            current_dk, current_dv = accumulate_prefix_gradients(
                ctx.kv_cache, dk, dv, ctx.offset
            )
        else:
            cached_dk = ctx.kv_cache.get('key_grad')
            cached_dv = ctx.kv_cache.get('value_grad')
            if cached_dk is not None:
                if cached_dv is None:
                    raise RuntimeError('Seq1F1B hybrid KV grad relay is missing value_grad')
                cached_len = cached_dk.size(1)
                if (
                    cached_dv.size(1) != cached_len
                    or cached_len > dk.size(1)
                    or cached_dk.shape[:1] != dk.shape[:1]
                    or cached_dv.shape[:1] != dv.shape[:1]
                    or cached_dk.shape[2:] != dk.shape[2:]
                    or cached_dv.shape[2:] != dv.shape[2:]
                ):
                    raise RuntimeError(
                        'Seq1F1B hybrid KV grad relay shape mismatch: '
                        f'cached={cached_dk.shape}/{cached_dv.shape}, current={dk.shape}/{dv.shape}'
                    )
                if cached_len > 0:
                    dk[:, :cached_len].add_(cached_dk)
                    dv[:, :cached_len].add_(cached_dv)

            if ctx.offset > 0:
                ctx.kv_cache['key_grad'] = dk[:, :ctx.offset].contiguous()
                ctx.kv_cache['value_grad'] = dv[:, :ctx.offset].contiguous()
            else:
                ctx.kv_cache.pop('key_grad', None)
                ctx.kv_cache.pop('value_grad', None)
            current_dk = dk[:, ctx.offset:ctx.seqlen_k].contiguous()
            current_dv = dv[:, ctx.offset:ctx.seqlen_k].contiguous()
        return dq, current_dk, current_dv, None, None, None, None


class Seq1F1BHybridSelfAttention(SelfAttention):
    seq1f1b_accepts_microbatch_cache_key = True

    """Causal softmax attention with Seq1F1B full-prefix KV relay.

    The trainer feeds one sequence chunk at a time. This module keeps the
    softmax layers semantically full-prefix causal by relaying prefix K/V in
    forward and prefix K/V gradients in backward, following StateFlow's
    Megatron implementation.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.causal,
        cp_comm_type: str = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
        )
        if not HAS_FLASH_ATTN:
            raise ImportError('Seq1F1B hybrid softmax attention requires flash-attn.')
        self.seq1f1b_kv_cache = {}
        self.seq1f1b_kv_cache_by_microbatch = {}

    def _clear_seq1f1b_kv_cache(self, clear_recompute_snapshots: bool = True) -> None:
        del clear_recompute_snapshots
        self.seq1f1b_kv_cache = {}
        self.seq1f1b_kv_cache_by_microbatch = {}

    def _full_recompute_seq1f1b_enabled(self, args, micro_sp_idx) -> bool:
        return (
            self.training
            and micro_sp_idx is not None
            and getattr(args, 'pipe_sp_splits', 1) > 1
            and getattr(args, 'recompute_granularity', None) == 'full'
        )

    @staticmethod
    def _seq1f1b_cache_key(split_context) -> int:
        return int(getattr(split_context, 'microbatch_key', 0))

    @staticmethod
    def _slice_rotary(rotary_pos_emb, start: int, end: int, seq_len: int):
        if rotary_pos_emb is None:
            return None
        if rotary_pos_emb.size(0) >= end:
            return rotary_pos_emb[start:end]
        return rotary_pos_emb[:seq_len]

    def _apply_seq1f1b_rotary(self, query, key, rotary_pos_emb, start: int, end: int):
        if rotary_pos_emb is None:
            return query, key
        if not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)

        q_pos_emb, k_pos_emb = rotary_pos_emb
        seq_len = query.size(0)
        q_pos_emb = self._slice_rotary(q_pos_emb, start, end, seq_len)
        k_pos_emb = self._slice_rotary(k_pos_emb, start, end, seq_len)
        if q_pos_emb is not None:
            query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config, cu_seqlens=None)
        if k_pos_emb is not None:
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=None)
        return query, key

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
        seq1f1b_microbatch_cache_key=None,
        seq1f1b_micro_sp_idx=None,
        seq1f1b_start=None,
        seq1f1b_end=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        split_context = get_seq_split_context()
        use_seq1f1b = split_context.pipe_sp_splits > 1
        if (
            not use_seq1f1b
            or key_value_states is not None
            or inference_context is not None
            or inference_params is not None
            or packed_seq_params is not None
            or rotary_pos_cos is not None
            or rotary_pos_sin is not None
            or attention_bias is not None
        ):
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                key_value_states=key_value_states,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
            )

        args = get_args()

        def _metadata_int(value, default=None):
            if value is None:
                return default
            if torch.is_tensor(value):
                value = value.item()
            value = int(value)
            return default if value < 0 else value

        micro_sp_idx = _metadata_int(seq1f1b_micro_sp_idx, split_context.micro_sp_idx)
        cache_key = self._seq1f1b_cache_key(split_context)
        cache_key = _metadata_int(seq1f1b_microbatch_cache_key, cache_key)
        start = _metadata_int(seq1f1b_start, split_context.start)
        start = 0 if start is None else start
        end = _metadata_int(seq1f1b_end, split_context.end)
        full_recompute_seq1f1b = self._full_recompute_seq1f1b_enabled(args, micro_sp_idx)

        if micro_sp_idx == 0 and not (full_recompute_seq1f1b and torch.is_grad_enabled()):
            self.seq1f1b_kv_cache_by_microbatch[cache_key] = {}
        kv_cache = self.seq1f1b_kv_cache_by_microbatch.setdefault(cache_key, {})

        if full_recompute_seq1f1b:
            if not torch.is_grad_enabled():
                cached_key = kv_cache.get('key')
                record_prefix_offset(
                    kv_cache,
                    micro_sp_idx,
                    0 if cached_key is None else cached_key.size(1),
                )
            else:
                restore_recompute_prefix(
                    kv_cache,
                    micro_sp_idx,
                    key_name='key',
                    value_name='value',
                    expected_offset=start,
                )

        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
        end = start + query.size(0) if end is None else end
        query, key = self._apply_seq1f1b_rotary(query, key, rotary_pos_emb, start, end)

        q = query.transpose(0, 1).contiguous()
        k = key.transpose(0, 1).contiguous()
        v = value.transpose(0, 1).contiguous()
        dropout_p = self.config.attention_dropout if self.training else 0.0
        context = Seq1F1BFlashAttnVarlenFunc.apply(
            q,
            k,
            v,
            kv_cache,
            dropout_p,
            True,
            None,
        )
        if (
            full_recompute_seq1f1b
            and not torch.is_grad_enabled()
            and micro_sp_idx == int(split_context.pipe_sp_splits) - 1
        ):
            seal_recompute_kv(kv_cache, key_name='key', value_name='value')
        context = context.transpose(0, 1).contiguous().view(query.size(0), query.size(1), -1)
        return self.linear_proj(context)
