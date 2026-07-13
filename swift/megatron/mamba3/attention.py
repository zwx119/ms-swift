# Copyright (c) 2026, Seq1F1B-Mamba3 Contributors.

"""Mamba-3 mixer adapter for Megatron Seq1F1B experiments.

This module keeps StateFlow's sequence chunking outside the kernel.  For SISO
Mamba-3, the official Triton autograd wrapper already accepts input states and
returns final states, so the adapter only threads those states between
Seq1F1B chunks.
"""

import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import make_viewless_tensor

from ..deltanet.context import get_seq_split_context

try:
    import triton
    from einops import rearrange
    from mamba_ssm.modules.mamba3 import Mamba3, heavy_tail_activation
    from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_bwd, angle_dt_fwd
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import (
        compute_ddt_dtrap_dinput_states,
        compute_dqktheta,
        compute_dqkv,
        compute_dzdo,
    )
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import (
        mamba3_siso_combined,
    )
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd
    HAS_MAMBA3 = True
    MAMBA3_IMPORT_ERROR = None
except Exception as exc:
    triton = None
    rearrange = None
    Mamba3 = None
    heavy_tail_activation = None
    angle_dt_bwd = None
    angle_dt_fwd = None
    compute_ddt_dtrap_dinput_states = None
    compute_dqktheta = None
    compute_dqkv = None
    compute_dzdo = None
    mamba3_siso_combined = None
    mamba3_siso_fwd = None
    HAS_MAMBA3 = False
    MAMBA3_IMPORT_ERROR = exc


def _mamba3_profile_enabled():
    return (
        os.environ.get("MAMBA3_ATTN_NVTX", "0") != "0"
        or os.environ.get("MAMBA3_LAYER_NVTX", "0") != "0"
    ) and torch.cuda.is_available()


@contextmanager
def _mamba3_profile(name):
    if not _mamba3_profile_enabled():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _detach_state_tuple(state):
    if state is None:
        return None
    return tuple(value.detach().clone() if torch.is_tensor(value) else value for value in state)


_MAMBA3_MEM_DEBUG_COUNTS = {}
_MAMBA3_ACTIVE_CTX_STATS = {}


def _mamba3_mem_debug_enabled():
    return (
        os.environ.get("MAMBA3_STATEFLOW_MEM_DEBUG", "0").lower()
        not in ("", "0", "false", "no")
        and torch.cuda.is_available()
    )


def _mamba3_bwd_intermediate_strategy():
    strategy = os.environ.get("MAMBA3_BWD_INTERMEDIATE_STRATEGY", "").strip().lower()
    if strategy in ("", "auto"):
        legacy = os.environ.get("MAMBA3_RECOMPUTE_BACKWARD_INTERMEDIATES")
        if legacy is not None:
            return "full" if legacy.strip().lower() not in ("", "0", "false", "no") else "save"
        return "save"
    if strategy in ("save", "saved", "none", "default"):
        return "save"
    if strategy in ("full", "recompute", "recompute_full"):
        return "full"
    if strategy in ("light", "partial"):
        return "light"
    raise RuntimeError(
        "MAMBA3_BWD_INTERMEDIATE_STRATEGY must be one of save, light, or full"
    )


def _mamba3_mem_debug_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def _mamba3_mem_debug_rank_allowed(rank):
    ranks = os.environ.get(
        "MAMBA3_STATEFLOW_MEM_DEBUG_RANKS",
        os.environ.get("MAMBA3_STATEFLOW_MEM_DEBUG_RANK", "0"),
    ).strip()
    if ranks.lower() in ("all", "*"):
        return True
    try:
        return rank in {int(item) for item in ranks.split(",") if item.strip()}
    except ValueError:
        return rank == 0


def _mamba3_tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    return 0


def _mamba3_tree_bytes(value, seen=None, debug_enabled=None):
    if debug_enabled is None:
        debug_enabled = _mamba3_mem_debug_enabled()
    if not debug_enabled:
        return 0
    if seen is None:
        seen = set()
    if torch.is_tensor(value):
        if value.device.type == "cuda":
            key = (value.device.index, value.data_ptr(), value.storage_offset())
            if key in seen:
                return 0
            seen.add(key)
        return _mamba3_tensor_bytes(value)
    if isinstance(value, dict):
        return sum(_mamba3_tree_bytes(v, seen, True) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_mamba3_tree_bytes(v, seen, True) for v in value)
    return 0


def _mamba3_gib(num_bytes):
    return float(num_bytes) / (1024.0 ** 3)


def _mamba3_active_ctx_update(state_cache, count_delta=0, bytes_delta=0):
    if not _mamba3_mem_debug_enabled():
        return {}
    rank = _mamba3_mem_debug_rank()
    if not _mamba3_mem_debug_rank_allowed(rank):
        return {}
    layer = None
    if isinstance(state_cache, dict):
        layer = state_cache.get("_debug_layer", None)

    def update_one(key):
        stats = _MAMBA3_ACTIVE_CTX_STATS.setdefault(
            key,
            {
                "count": 0,
                "bytes": 0,
                "max_count": 0,
                "max_bytes": 0,
            },
        )
        stats["count"] = max(0, stats["count"] + count_delta)
        stats["bytes"] = max(0, stats["bytes"] + bytes_delta)
        stats["max_count"] = max(stats["max_count"], stats["count"])
        stats["max_bytes"] = max(stats["max_bytes"], stats["bytes"])
        return stats

    rank_stats = update_one((rank, "rank"))
    fields = {
        "active_ctx_count_rank": rank_stats["count"],
        "active_ctx_rank_bytes": rank_stats["bytes"],
        "max_active_ctx_count_rank": rank_stats["max_count"],
        "max_active_ctx_rank_bytes": rank_stats["max_bytes"],
    }
    if layer is not None:
        layer_stats = update_one((rank, "layer", layer))
        fields.update(
            {
                "active_ctx_count_layer": layer_stats["count"],
                "active_ctx_layer_bytes": layer_stats["bytes"],
                "max_active_ctx_count_layer": layer_stats["max_count"],
                "max_active_ctx_layer_bytes": layer_stats["max_bytes"],
            }
        )
    return fields


def _mamba3_mem_debug_log(tag, state_cache=None, **fields):
    if not _mamba3_mem_debug_enabled():
        return
    rank = _mamba3_mem_debug_rank()
    if not _mamba3_mem_debug_rank_allowed(rank):
        return
    layer = None
    micro_sp = None
    cache_key = None
    if isinstance(state_cache, dict):
        layer = state_cache.get("_debug_layer", None)
        micro_sp = state_cache.get("_debug_micro_sp_idx", None)
        cache_key = state_cache.get("_debug_cache_key", None)
    layer_filter = os.environ.get("MAMBA3_STATEFLOW_MEM_DEBUG_LAYERS", "").strip()
    if layer_filter and layer is not None:
        allowed = {item.strip() for item in layer_filter.split(",") if item.strip()}
        if str(layer) not in allowed:
            return
    limit = int(os.environ.get("MAMBA3_STATEFLOW_MEM_DEBUG_LIMIT_PER_TAG", "200"))
    count_key = (rank, tag, layer)
    count = _MAMBA3_MEM_DEBUG_COUNTS.get(count_key, 0)
    if count >= limit:
        return
    _MAMBA3_MEM_DEBUG_COUNTS[count_key] = count + 1
    if os.environ.get("MAMBA3_STATEFLOW_MEM_DEBUG_SYNC", "0") not in ("", "0"):
        torch.cuda.synchronize()
    device = torch.cuda.current_device()
    parts = [
        "[M3MEM]",
        f"tag={tag}",
        f"rank={rank}",
        f"dev={device}",
        f"layer={layer}",
        f"micro_sp={micro_sp}",
        f"cache_key={cache_key}",
        f"alloc_gb={_mamba3_gib(torch.cuda.memory_allocated(device)):.3f}",
        f"reserved_gb={_mamba3_gib(torch.cuda.memory_reserved(device)):.3f}",
        f"max_alloc_gb={_mamba3_gib(torch.cuda.max_memory_allocated(device)):.3f}",
    ]
    for key, value in fields.items():
        if key.endswith("_bytes"):
            parts.append(f"{key}_gb={_mamba3_gib(value):.3f}")
        else:
            parts.append(f"{key}={value}")
    print(" ".join(parts), flush=True)


class Mamba3StateRelayFunc(torch.autograd.Function):
    """Direct Mamba-3 SISO fwd/bwd with GDN-style state relay.

    Seq1F1B chunk state is read/written through `state_cache`, not passed as a
    tensor argument to `.apply()`. Backward relays gradients for final states via
    `state_cache["d_state"]`, matching DeltaNet/GDN's cross-chunk contract.
    """

    @staticmethod
    def _set_allocator():
        if triton is None:
            return
        try:
            triton.set_allocator(Mamba3StateRelayFunc._triton_alloc_fn)
        except Exception:
            pass

    @staticmethod
    def _triton_alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    @staticmethod
    def _cast_grad(grad, dtype):
        if grad is None or dtype is None:
            return grad
        if torch.is_tensor(grad) and grad.dtype != dtype:
            return grad.to(dtype)
        return grad

    @staticmethod
    def forward(
        ctx,
        Q,
        K,
        V,
        ADT,
        DT,
        Trap,
        Q_bias,
        K_bias,
        Angles,
        D,
        Z,
        state_cache,
        chunk_size,
        track_grad,
    ):
        Mamba3StateRelayFunc._set_allocator()

        input_dtypes = (
            Q.dtype,
            K.dtype,
            V.dtype,
            ADT.dtype,
            DT.dtype,
            Trap.dtype,
            Q_bias.dtype,
            K_bias.dtype,
            Angles.dtype,
            D.dtype if D is not None else None,
            Z.dtype if Z is not None else None,
        )

        Q = Q.to(torch.bfloat16)
        K = K.to(torch.bfloat16)
        V = V.to(torch.bfloat16)
        Trap = Trap.to(torch.bfloat16)
        Angles = Angles.to(torch.bfloat16)
        if Z is not None:
            Z = Z.to(torch.bfloat16)

        recompute_replay = bool(state_cache.get("_recompute_replay", False))
        if recompute_replay:
            initial_state = state_cache.get("_recompute_initial_state", None)
        else:
            initial_state = state_cache.get("recurrent_state", None)

        if initial_state is None:
            Input_Angle_State = None
            Input_SSM_State = None
            Input_K_State = None
            Input_V_State = None
        else:
            if len(initial_state) != 4:
                raise RuntimeError("Mamba3 recurrent state cache must hold a 4-tuple")
            Input_Angle_State, Input_SSM_State, Input_K_State, Input_V_State = (
                value.detach() if torch.is_tensor(value) else value
                for value in initial_state
            )

        all_states_present = (
            Input_Angle_State is not None
            and Input_SSM_State is not None
            and Input_K_State is not None
            and Input_V_State is not None
        )
        all_states_absent = (
            Input_Angle_State is None
            and Input_SSM_State is None
            and Input_K_State is None
            and Input_V_State is None
        )
        if not (all_states_present or all_states_absent):
            raise RuntimeError("Mamba3 input states must be provided together")

        needs_backward = bool(track_grad) and any(ctx.needs_input_grad[:11])
        _mamba3_mem_debug_log(
            "relay_fwd_pre",
            state_cache,
            needs_backward=needs_backward,
            recompute_replay=recompute_replay,
            q_bytes=_mamba3_tree_bytes(Q),
            k_bytes=_mamba3_tree_bytes(K),
            v_bytes=_mamba3_tree_bytes(V),
            adt_bytes=_mamba3_tree_bytes(ADT),
            z_bytes=_mamba3_tree_bytes(Z),
            input_state_bytes=_mamba3_tree_bytes(initial_state),
            state_cache_bytes=_mamba3_tree_bytes(state_cache),
            seqlen=Q.shape[1],
            nheads=V.shape[2],
        )
        Angles_Cumsum, Final_Angle_State = angle_dt_fwd(
            Angles,
            DT,
            init_state=Input_Angle_State,
            chunk_size=chunk_size,
            return_output_state=True,
            cu_seqlens=None,
        )
        kernel_input_states = (
            (Input_SSM_State, Input_K_State, Input_V_State)
            if Input_SSM_State is not None
            else None
        )
        (
            Out,
            Out_v,
            SSM_States,
            DA_CS,
            DA_CS_SUM,
            Q_rot,
            K_scaled,
            QK_dot,
            Scale,
            Gamma,
            Final_States,
        ) = mamba3_siso_fwd(
            Q,
            K,
            V,
            ADT,
            DT,
            Trap,
            Q_bias,
            K_bias,
            Angles_Cumsum,
            D,
            Z,
            kernel_input_states,
            chunk_size=chunk_size,
            store_states_adt_outv=needs_backward,
            return_final_states=True,
            cu_seqlens=None,
        )
        Final_SSM_State, Final_K_State, Final_V_State = Final_States
        final_state = (Final_Angle_State, Final_SSM_State, Final_K_State, Final_V_State)
        if not recompute_replay:
            state_cache["recurrent_state"] = _detach_state_tuple(final_state)

        _mamba3_mem_debug_log(
            "relay_fwd_post",
            state_cache,
            needs_backward=needs_backward,
            recompute_replay=recompute_replay,
            out_bytes=_mamba3_tree_bytes(Out),
            out_v_bytes=_mamba3_tree_bytes(Out_v),
            ssm_states_bytes=_mamba3_tree_bytes(SSM_States),
            q_store_bytes=_mamba3_tree_bytes(Q_rot),
            k_store_bytes=_mamba3_tree_bytes(K_scaled),
            qk_store_bytes=_mamba3_tree_bytes(QK_dot),
            scale_gamma_bytes=_mamba3_tree_bytes((Scale, Gamma)),
            decay_store_bytes=_mamba3_tree_bytes((DA_CS, DA_CS_SUM)),
            final_state_bytes=_mamba3_tree_bytes(final_state),
            state_cache_bytes=_mamba3_tree_bytes(state_cache),
        )

        ctx._state_cache = state_cache
        ctx._needs_backward = needs_backward
        ctx.chunk_size = chunk_size
        ctx.has_D = D is not None
        ctx.has_Z = Z is not None
        ctx.has_input_state = Input_SSM_State is not None
        ctx.input_dtypes = input_dtypes
        ctx.bwd_intermediate_strategy = (
            _mamba3_bwd_intermediate_strategy() if needs_backward else "save"
        )
        ctx.recompute_bwd_intermediates = (
            ctx.bwd_intermediate_strategy == "full" and needs_backward
        )
        _mamba3_mem_debug_log(
            "relay_fwd_strategy",
            state_cache,
            bwd_intermediate_strategy=ctx.bwd_intermediate_strategy,
            recompute_bwd_intermediates=ctx.recompute_bwd_intermediates,
        )

        if needs_backward:
            empty = torch.empty((), device=Q.device)
            if ctx.recompute_bwd_intermediates:
                saved_tensors = (
                    Q,
                    K,
                    V,
                    ADT,
                    DT,
                    Trap,
                    Q_bias,
                    K_bias,
                    Angles,
                    D if D is not None else empty,
                    Z if Z is not None else empty,
                    Input_Angle_State if Input_Angle_State is not None else empty,
                    Input_SSM_State if Input_SSM_State is not None else empty,
                    Input_K_State if Input_K_State is not None else empty,
                    Input_V_State if Input_V_State is not None else empty,
                )
            elif ctx.bwd_intermediate_strategy == "light":
                saved_tensors = (
                    Q,
                    K,
                    V,
                    ADT,
                    DT,
                    Trap,
                    Q_bias,
                    K_bias,
                    Angles,
                    D if D is not None else empty,
                    Z if Z is not None else empty,
                    Input_Angle_State if Input_Angle_State is not None else empty,
                    Input_SSM_State if Input_SSM_State is not None else empty,
                    Input_K_State if Input_K_State is not None else empty,
                    Input_V_State if Input_V_State is not None else empty,
                    Out_v,
                    SSM_States,
                    DA_CS,
                    DA_CS_SUM,
                    Q_rot,
                    K_scaled,
                    QK_dot,
                    Scale,
                    Gamma,
                )
            else:
                saved_tensors = (
                    Q,
                    K,
                    V,
                    ADT,
                    DT,
                    Trap,
                    Q_bias,
                    K_bias,
                    Angles,
                    Angles_Cumsum,
                    D if D is not None else empty,
                    Z if Z is not None else empty,
                    Input_SSM_State if Input_SSM_State is not None else empty,
                    Input_K_State if Input_K_State is not None else empty,
                    Input_V_State if Input_V_State is not None else empty,
                    Out,
                    Out_v,
                    SSM_States,
                    DA_CS,
                    DA_CS_SUM,
                    Q_rot,
                    K_scaled,
                    QK_dot,
                    Scale,
                    Gamma,
                    Final_SSM_State,
                )
            ctx.save_for_backward(*saved_tensors)
            if _mamba3_mem_debug_enabled():
                ctx._mamba3_saved_tensor_bytes = _mamba3_tree_bytes(saved_tensors, debug_enabled=True)
                ctx._mamba3_active_ctx_recorded = True
                _mamba3_mem_debug_log(
                    "relay_ctx_save",
                    state_cache,
                    saved_tensor_bytes=ctx._mamba3_saved_tensor_bytes,
                    **_mamba3_active_ctx_update(
                        state_cache, 1, ctx._mamba3_saved_tensor_bytes
                    ),
                )
            else:
                ctx._mamba3_saved_tensor_bytes = 0
                ctx._mamba3_active_ctx_recorded = False
        else:
            ctx.save_for_backward()
            ctx._mamba3_saved_tensor_bytes = 0
            ctx._mamba3_active_ctx_recorded = False

        return Out

    @staticmethod
    def backward(ctx, grad_out):
        if not ctx._needs_backward:
            return (None,) * 14

        Mamba3StateRelayFunc._set_allocator()
        state_cache = ctx._state_cache
        grad_state = state_cache.pop("d_state", None)
        if grad_state is None:
            grad_final_angle_state = None
            grad_final_ssm_state = None
            grad_final_k_state = None
            grad_final_v_state = None
        else:
            if len(grad_state) != 4:
                raise RuntimeError("Mamba3 d_state cache must hold a 4-tuple")
            (
                grad_final_angle_state,
                grad_final_ssm_state,
                grad_final_k_state,
                grad_final_v_state,
            ) = grad_state

        _mamba3_mem_debug_log(
            "relay_bwd_pre",
            state_cache,
            grad_out_bytes=_mamba3_tree_bytes(grad_out),
            grad_state_bytes=_mamba3_tree_bytes(grad_state),
            saved_tensor_bytes=_mamba3_tree_bytes(ctx.saved_tensors),
            has_input_state=ctx.has_input_state,
            bwd_intermediate_strategy=ctx.bwd_intermediate_strategy,
            **_mamba3_active_ctx_update(state_cache, 0, 0),
        )

        if ctx.recompute_bwd_intermediates:
            (
                Q,
                K,
                V,
                ADT,
                DT,
                Trap,
                Q_bias,
                K_bias,
                Angles,
                D_save,
                Z_save,
                Input_Angle_State_save,
                Input_SSM_State_save,
                Input_K_State_save,
                Input_V_State_save,
            ) = ctx.saved_tensors
            D = D_save if ctx.has_D else None
            Z = Z_save if ctx.has_Z else None
            Input_Angle_State = Input_Angle_State_save if ctx.has_input_state else None
            Input_SSM_State = Input_SSM_State_save if ctx.has_input_state else None
            Input_K_State = Input_K_State_save if ctx.has_input_state else None
            Input_V_State = Input_V_State_save if ctx.has_input_state else None
            with torch.no_grad():
                Angles_Cumsum, _Final_Angle_State = angle_dt_fwd(
                    Angles,
                    DT,
                    init_state=Input_Angle_State,
                    chunk_size=ctx.chunk_size,
                    return_output_state=True,
                    cu_seqlens=None,
                )
                kernel_input_states = (
                    (Input_SSM_State, Input_K_State, Input_V_State)
                    if Input_SSM_State is not None
                    else None
                )
                (
                    Out,
                    Out_v,
                    SSM_States,
                    DA_CS,
                    DA_CS_SUM,
                    Q_rot,
                    K_scaled,
                    QK_dot,
                    Scale,
                    Gamma,
                    _Final_States,
                ) = mamba3_siso_fwd(
                    Q,
                    K,
                    V,
                    ADT,
                    DT,
                    Trap,
                    Q_bias,
                    K_bias,
                    Angles_Cumsum,
                    D,
                    Z,
                    kernel_input_states,
                    chunk_size=ctx.chunk_size,
                    store_states_adt_outv=True,
                    return_final_states=False,
                    cu_seqlens=None,
                )
            _mamba3_mem_debug_log(
                "relay_bwd_recompute_post",
                state_cache,
                out_bytes=_mamba3_tree_bytes(Out),
                out_v_bytes=_mamba3_tree_bytes(Out_v),
                ssm_states_bytes=_mamba3_tree_bytes(SSM_States),
                q_store_bytes=_mamba3_tree_bytes(Q_rot),
                k_store_bytes=_mamba3_tree_bytes(K_scaled),
                qk_store_bytes=_mamba3_tree_bytes(QK_dot),
                scale_gamma_bytes=_mamba3_tree_bytes((Scale, Gamma)),
                decay_store_bytes=_mamba3_tree_bytes((DA_CS, DA_CS_SUM)),
            )
        elif ctx.bwd_intermediate_strategy == "light":
            (
                Q,
                K,
                V,
                ADT,
                DT,
                Trap,
                Q_bias,
                K_bias,
                Angles,
                D_save,
                Z_save,
                Input_Angle_State_save,
                Input_SSM_State_save,
                Input_K_State_save,
                Input_V_State_save,
                Out_v,
                SSM_States,
                DA_CS,
                DA_CS_SUM,
                Q_rot,
                K_scaled,
                QK_dot,
                Scale,
                Gamma,
            ) = ctx.saved_tensors
            D = D_save if ctx.has_D else None
            Z = Z_save if ctx.has_Z else None
            Input_Angle_State = Input_Angle_State_save if ctx.has_input_state else None
            Input_K_State = Input_K_State_save if ctx.has_input_state else None
            Input_V_State = Input_V_State_save if ctx.has_input_state else None
            with torch.no_grad():
                Angles_Cumsum, _Final_Angle_State = angle_dt_fwd(
                    Angles,
                    DT,
                    init_state=Input_Angle_State,
                    chunk_size=ctx.chunk_size,
                    return_output_state=True,
                    cu_seqlens=None,
                )
            _mamba3_mem_debug_log(
                "relay_bwd_light_post",
                state_cache,
                angles_cumsum_bytes=_mamba3_tree_bytes(Angles_Cumsum),
            )
        else:
            (
                Q,
                K,
                V,
                ADT,
                DT,
                Trap,
                Q_bias,
                K_bias,
                Angles,
                Angles_Cumsum,
                D_save,
                Z_save,
                Input_SSM_State_save,
                Input_K_State_save,
                Input_V_State_save,
                Out,
                Out_v,
                SSM_States,
                DA_CS,
                DA_CS_SUM,
                Q_rot,
                K_scaled,
                QK_dot,
                Scale,
                Gamma,
                _Final_SSM_State_save,
            ) = ctx.saved_tensors
            D = D_save if ctx.has_D else None
            Z = Z_save if ctx.has_Z else None
            Input_K_State = Input_K_State_save if ctx.has_input_state else None
            Input_V_State = Input_V_State_save if ctx.has_input_state else None
        if grad_out is None:
            grad_out = torch.zeros_like(V)

        if Z is not None:
            dZ, grad_out_scaled = compute_dzdo(
                grad_out, Z, Out_v, chunk_size=ctx.chunk_size
            )
        else:
            dZ = None
            grad_out_scaled = grad_out

        dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, dInput_SSM_State = compute_dqkv(
            q=Q_rot,
            k=K_scaled,
            v=V,
            da_cs=DA_CS,
            da_cs_sum=DA_CS_SUM,
            qk_dot=QK_dot,
            SSM_States=SSM_States,
            do=grad_out_scaled,
            d_ossm_state=grad_final_ssm_state,
            d_ov_state=grad_final_v_state,
            D=D,
            chunk_size=ctx.chunk_size,
            has_input_state=ctx.has_input_state,
            Cu_Seqlens=None,
        )

        dQ, dK, dQ_bias, dK_bias, dAngles_Cumsum, dScale, dGamma = compute_dqktheta(
            q=Q,
            k=K,
            scale=Scale,
            gamma=Gamma,
            q_bias=Q_bias,
            k_bias=K_bias,
            angles=Angles_Cumsum,
            dq_in=dQ_mid,
            dk_in=dK_mid,
            dqk=dQK_dot,
            d_ok_state=grad_final_k_state,
            chunk_size=ctx.chunk_size,
            Cu_Seqlens=None,
        )

        dDT, dTrap, dInput_SSM_State_final, dInput_K_State, dInput_V_State = (
            compute_ddt_dtrap_dinput_states(
                dscale=dScale,
                dgamma=dGamma,
                dt=DT,
                trap=Trap.float(),
                d_issm_state=dInput_SSM_State if ctx.has_input_state else None,
                input_k_state=Input_K_State,
                input_v_state=Input_V_State,
                Cu_Seqlens=None,
            )
        )

        dAngles, dDT_angle, dInput_Angle_State = angle_dt_bwd(
            grad_out=dAngles_Cumsum,
            angle=Angles,
            dt=DT,
            has_init_state=ctx.has_input_state,
            chunk_size=ctx.chunk_size,
            grad_output_state=grad_final_angle_state,
            cu_seqlens=None,
        )
        dDT = dDT + dDT_angle

        if ctx.has_input_state:
            dInput_SSM_State = dInput_SSM_State_final
            state_grads = (
                dInput_Angle_State,
                dInput_SSM_State,
                dInput_K_State,
                dInput_V_State,
            )
            if any(grad is not None for grad in state_grads):
                state_cache["d_state"] = _detach_state_tuple(state_grads)

        _mamba3_mem_debug_log(
            "relay_bwd_post",
            state_cache,
            dqkv_bytes=_mamba3_tree_bytes((dQ, dK, dV)),
            ddt_trap_bytes=_mamba3_tree_bytes((dDT, dTrap, dADT)),
            dparam_bytes=_mamba3_tree_bytes((dQ_bias, dK_bias, dD, dZ, dAngles)),
            d_state_out_bytes=_mamba3_tree_bytes(state_cache.get("d_state", None)),
            state_cache_bytes=_mamba3_tree_bytes(state_cache),
        )
        if getattr(ctx, "_mamba3_active_ctx_recorded", False):
            _mamba3_mem_debug_log(
                "relay_ctx_release",
                state_cache,
                saved_tensor_bytes=getattr(ctx, "_mamba3_saved_tensor_bytes", 0),
                **_mamba3_active_ctx_update(
                    state_cache,
                    -1,
                    -getattr(ctx, "_mamba3_saved_tensor_bytes", 0),
                ),
            )

        (
            q_dtype,
            k_dtype,
            v_dtype,
            adt_dtype,
            dt_dtype,
            trap_dtype,
            q_bias_dtype,
            k_bias_dtype,
            angles_dtype,
            d_dtype,
            z_dtype,
        ) = ctx.input_dtypes

        return (
            Mamba3StateRelayFunc._cast_grad(dQ, q_dtype),
            Mamba3StateRelayFunc._cast_grad(dK, k_dtype),
            Mamba3StateRelayFunc._cast_grad(dV, v_dtype),
            Mamba3StateRelayFunc._cast_grad(dADT, adt_dtype),
            Mamba3StateRelayFunc._cast_grad(dDT, dt_dtype),
            Mamba3StateRelayFunc._cast_grad(dTrap, trap_dtype),
            Mamba3StateRelayFunc._cast_grad(dQ_bias, q_bias_dtype),
            Mamba3StateRelayFunc._cast_grad(dK_bias, k_bias_dtype),
            Mamba3StateRelayFunc._cast_grad(dAngles, angles_dtype),
            Mamba3StateRelayFunc._cast_grad(dD, d_dtype),
            Mamba3StateRelayFunc._cast_grad(dZ, z_dtype),
            None,
            None,
            None,
        )


class Mamba3Attention(MegatronModule):
    seq1f1b_accepts_microbatch_cache_key = True

    """Drop-in recurrent mixer for MCore TransformerLayer self-attention."""

    def __init__(
        self,
        config,
        submodules=None,
        layer_number=1,
        attn_mask_type=None,
        cp_comm_type=None,
    ):
        del submodules, attn_mask_type, cp_comm_type
        super().__init__(config=config)
        if not HAS_MAMBA3:
            raise RuntimeError(
                "--use-mamba3 requires the official mamba_ssm Mamba-3 package. "
                "Set MAMBA3_DIR=/path/to/state-spaces/mamba or install it, "
                "and run on a CUDA-visible worker for Triton kernels."
            ) from MAMBA3_IMPORT_ERROR
        if rearrange is None:
            raise RuntimeError("--use-mamba3 requires einops")

        args = get_args()
        self.layer_number = max(1, layer_number)
        self.hidden_size = config.hidden_size
        self.sequence_parallel = config.sequence_parallel

        if parallel_state.get_tensor_model_parallel_world_size() != 1:
            raise RuntimeError(
                "Mamba3Attention currently supports tensor-model-parallel-size=1. "
                "The official Mamba-3 module uses regular nn.Linear weights; "
                "TP sharding needs a separate parameter-partitioned adapter."
            )
        if self.sequence_parallel:
            raise RuntimeError(
                "Mamba3Attention currently requires Megatron sequence_parallel=False. "
                "StateFlow pipe-sp splitting is still supported via --pipe-sp-splits."
            )

        self.mixer = Mamba3(
            d_model=config.hidden_size,
            d_state=getattr(args, "mamba3_d_state", 128),
            expand=getattr(args, "mamba3_expand", 2),
            headdim=getattr(args, "mamba3_head_dim", 64),
            ngroups=getattr(args, "mamba3_ngroups", 1),
            rope_fraction=getattr(args, "mamba3_rope_fraction", 0.5),
            dt_min=getattr(args, "mamba3_dt_min", 0.001),
            dt_max=getattr(args, "mamba3_dt_max", 0.1),
            dt_init_floor=getattr(args, "mamba3_dt_init_floor", 1e-4),
            A_floor=getattr(args, "mamba3_a_floor", 1e-4),
            is_outproj_norm=getattr(args, "mamba3_outproj_norm", False),
            is_mimo=getattr(args, "mamba3_is_mimo", False),
            mimo_rank=getattr(args, "mamba3_mimo_rank", 4),
            fuse_pregate_headwise_norm=getattr(
                args, "mamba3_fuse_pregate_headwise_norm", True
            ),
            chunk_size=getattr(args, "mamba3_chunk_size", 64),
            layer_idx=self.layer_number,
        )

        # Per-layer recurrent state relay, keyed by original microbatch when
        # the schedule exposes one. Each value is a dict containing detached
        # forward states and backward d_state for that microbatch.
        self.state_cache = {}
        self.recompute_state_init_cache = {}

    def _cache_key(self, microbatch_cache_key):
        if microbatch_cache_key is None:
            return 0
        if torch.is_tensor(microbatch_cache_key):
            microbatch_cache_key = microbatch_cache_key.item()
        return int(microbatch_cache_key)

    def _state_cache_for_key(self, cache_key):
        return self.state_cache.setdefault(cache_key, {})

    def _clear_states(self, cache_key=None, clear_recompute_snapshots=True):
        if cache_key is None:
            self.state_cache = {}
        else:
            self.state_cache.pop(cache_key, None)
        if clear_recompute_snapshots:
            self.recompute_state_init_cache = {}

    def _full_recompute_seq1f1b_enabled(self, args, micro_sp_idx):
        return (
            self.training
            and micro_sp_idx is not None
            and getattr(args, "pipe_sp_splits", 1) > 1
            and getattr(args, "recompute_granularity", None) == "full"
        )

    def _snapshot_recompute_value(self, value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, tuple):
            return tuple(self._snapshot_recompute_value(item) for item in value)
        return value

    def _begin_recompute_safe_state(self, args, cache_key, state_cache, micro_sp_idx):
        if not self._full_recompute_seq1f1b_enabled(args, micro_sp_idx):
            return False

        snapshot_key = (cache_key, micro_sp_idx)
        if not torch.is_grad_enabled():
            snapshot = self._snapshot_recompute_value(
                state_cache.get("recurrent_state", None)
            )
            state_stack = self.recompute_state_init_cache.setdefault(snapshot_key, [])
            state_stack.append(snapshot)
            _mamba3_mem_debug_log(
                "recompute_snapshot",
                state_cache,
                snapshot_bytes=_mamba3_tree_bytes(snapshot),
                snapshot_stack_len=len(state_stack),
                recompute_cache_bytes=_mamba3_tree_bytes(self.recompute_state_init_cache),
            )
            return False

        state_stack = self.recompute_state_init_cache.get(snapshot_key)
        if state_stack is None or len(state_stack) == 0:
            raise RuntimeError(
                "Missing Mamba3 recompute state snapshot for "
                f"cache_key={cache_key}, micro_sp_idx={micro_sp_idx}"
            )
        state_cache["_recompute_replay"] = True
        state_cache["_recompute_initial_state"] = state_stack.pop()
        _mamba3_mem_debug_log(
            "recompute_replay_begin",
            state_cache,
            replay_state_bytes=_mamba3_tree_bytes(state_cache.get("_recompute_initial_state", None)),
            remaining_snapshot_stack_len=len(state_stack),
            recompute_cache_bytes=_mamba3_tree_bytes(self.recompute_state_init_cache),
        )
        return True

    def _end_recompute_safe_state(self, state_cache, recompute_replay):
        if not recompute_replay:
            return
        state_cache.pop("_recompute_replay", None)
        state_cache.pop("_recompute_initial_state", None)

    def _mamba3_siso(
        self, u, initial_state=None, return_final_state=False, state_cache=None
    ):
        """Run official Mamba-3 SISO prefill while exposing input/final states."""
        m = self.mixer
        batch, seqlen, _ = u.shape
        if seqlen <= 0:
            raise RuntimeError("Mamba3Attention received an empty sequence chunk")

        zxBCdtAtrap = m.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                m.d_inner,
                m.d_inner,
                m.d_state * m.num_bc_heads * m.mimo_rank,
                m.d_state * m.num_bc_heads * m.mimo_rank,
                m.nheads,
                m.nheads,
                m.nheads,
                m.num_rope_angles,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=m.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=m.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=m.mimo_rank, g=m.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=m.mimo_rank, g=m.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        A = -heavy_tail_activation(dd_A.to(torch.float32))
        A = torch.clamp(A, max=-m.A_floor)
        dt = F.softplus(dd_dt + m.dt_bias)
        adt = rearrange(A * dt, "b l n -> b n l")
        dt = rearrange(dt, "b l n -> b n l")

        angles = angles.unsqueeze(-2).expand(-1, -1, m.nheads, -1).to(torch.float32)

        B = m.B_norm(B)
        C = m.C_norm(C)

        final_state = None
        if state_cache is not None:
            y = Mamba3StateRelayFunc.apply(
                C.squeeze(2),
                B.squeeze(2),
                x,
                adt,
                dt,
                trap,
                m.C_bias.squeeze(1),
                m.B_bias.squeeze(1),
                angles,
                m.D,
                z if not m.is_outproj_norm else None,
                state_cache,
                m.chunk_size,
                torch.is_grad_enabled(),
            )
        else:
            if _mamba3_mem_debug_enabled():
                debug_cache = {
                    "_debug_layer": self.layer_number,
                    "_debug_micro_sp_idx": None,
                    "_debug_cache_key": None,
                }
                _mamba3_mem_debug_log(
                    "official_fwd_pre",
                    debug_cache,
                    needs_backward=torch.is_grad_enabled()
                    and any(
                        tensor.requires_grad
                        for tensor in (
                            C,
                            B,
                            x,
                            adt,
                            dt,
                            trap,
                            m.C_bias,
                            m.B_bias,
                            angles,
                            m.D,
                            z if not m.is_outproj_norm else None,
                        )
                        if torch.is_tensor(tensor)
                    ),
                    q_bytes=_mamba3_tree_bytes(C.squeeze(2)),
                    k_bytes=_mamba3_tree_bytes(B.squeeze(2)),
                    v_bytes=_mamba3_tree_bytes(x),
                    adt_bytes=_mamba3_tree_bytes(adt),
                    z_bytes=_mamba3_tree_bytes(z if not m.is_outproj_norm else None),
                    seqlen=u.shape[1],
                    nheads=x.shape[2],
                )
            y = mamba3_siso_combined(
                Q=C.squeeze(2),
                K=B.squeeze(2),
                V=x,
                ADT=adt,
                DT=dt,
                Trap=trap,
                Q_bias=m.C_bias.squeeze(1),
                K_bias=m.B_bias.squeeze(1),
                Angles=angles,
                D=m.D,
                Z=z if not m.is_outproj_norm else None,
                chunk_size=m.chunk_size,
                Input_States=initial_state,
                return_final_states=return_final_state,
                cu_seqlens=None,
            )

            if return_final_state:
                y, last_angle, last_ssm, last_k, last_v = y
                final_state = (last_angle, last_ssm, last_k, last_v)
            if _mamba3_mem_debug_enabled():
                _mamba3_mem_debug_log(
                    "official_fwd_post",
                    debug_cache,
                    out_bytes=_mamba3_tree_bytes(y),
                    final_state_bytes=_mamba3_tree_bytes(final_state),
                    seqlen=u.shape[1],
                    nheads=x.shape[2],
                )

        y = rearrange(y, "b l h p -> b l (h p)")
        if m.is_outproj_norm:
            z = rearrange(z, "b l h p -> b l (h p)")
            y = m.norm(y, z)

        out = m.out_proj(y.to(x.dtype))
        return out, final_state

    def _mamba3_mimo_or_official(self, u, return_final_state):
        if return_final_state:
            raise RuntimeError(
                "Mamba3 MIMO/TileLang path is only wired for SP=1 prefill. "
                "Seq1F1B chunk state passing currently uses the SISO Triton path."
            )
        return self.mixer(u), None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        seq1f1b_microbatch_cache_key=None,
        seq1f1b_micro_sp_idx=None,
        seq1f1b_start=None,
        seq1f1b_end=None,
    ):
        del attention_mask, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin
        del attention_bias, packed_seq_params, sequence_len_offset
        del seq1f1b_start, seq1f1b_end
        if key_value_states is not None:
            raise RuntimeError("Mamba3Attention only supports decoder-only self attention")
        if inference_params is None:
            inference_params = inference_context

        args = get_args()
        split_context = get_seq_split_context()

        def _metadata_int(value, default=None):
            if value is None:
                return default
            if torch.is_tensor(value):
                value = value.item()
            value = int(value)
            return default if value < 0 else value

        micro_sp_idx = _metadata_int(seq1f1b_micro_sp_idx, split_context.micro_sp_idx)
        microbatch_cache_key = _metadata_int(
            seq1f1b_microbatch_cache_key,
            getattr(split_context, "microbatch_key", 0),
        )
        if getattr(args, "pipe_sp_splits", 1) <= 1:
            micro_sp_idx = None

        with _mamba3_profile("Mamba3Attention.forward"):
            if inference_params is not None:
                u = hidden_states.transpose(0, 1).contiguous()
                out = self.mixer(u, inference_params=inference_params)
                return out.transpose(0, 1).contiguous(), None

            cache_key = self._cache_key(microbatch_cache_key)
            return_final_state = (
                micro_sp_idx is not None and getattr(args, "pipe_sp_splits", 1) > 1
            )
            state_cache = None
            initial_state = None
            recompute_replay = False
            if return_final_state:
                full_recompute_seq1f1b = self._full_recompute_seq1f1b_enabled(
                    args, micro_sp_idx
                )
                if micro_sp_idx == 0 and not (
                    full_recompute_seq1f1b and torch.is_grad_enabled()
                ):
                    self._clear_states(
                        cache_key, clear_recompute_snapshots=not full_recompute_seq1f1b
                    )
                state_cache = self._state_cache_for_key(cache_key)
                if _mamba3_mem_debug_enabled():
                    state_cache["_debug_layer"] = self.layer_number
                    state_cache["_debug_micro_sp_idx"] = micro_sp_idx
                    state_cache["_debug_cache_key"] = cache_key
                recompute_replay = self._begin_recompute_safe_state(
                    args, cache_key, state_cache, micro_sp_idx
                )
                if recompute_replay:
                    initial_state = state_cache.get("_recompute_initial_state", None)
                else:
                    initial_state = state_cache.get("recurrent_state", None)
                if micro_sp_idx > 0 and initial_state is None:
                    raise RuntimeError(
                        "Missing Mamba3 Seq1F1B recurrent state for "
                        f"micro_sp_idx={micro_sp_idx}, cache_key={cache_key}. "
                        "This usually means chunks from different microbatches were "
                        "interleaved without a microbatch cache key."
                    )

            try:
                u = hidden_states.transpose(0, 1).contiguous()
                if self.mixer.is_mimo:
                    out, final_state = self._mamba3_mimo_or_official(u, return_final_state)
                else:
                    out, final_state = self._mamba3_siso(
                        u,
                        initial_state=initial_state,
                        return_final_state=return_final_state,
                        state_cache=state_cache if return_final_state else None,
                    )

                if return_final_state and final_state is not None:
                    state_cache["recurrent_state"] = _detach_state_tuple(final_state)

                output = out.transpose(0, 1).contiguous()
                output = make_viewless_tensor(
                    inp=output,
                    requires_grad=output.requires_grad,
                    keep_graph=True,
                )
                return output, None
            finally:
                if state_cache is not None:
                    self._end_recompute_safe_state(state_cache, recompute_replay)
