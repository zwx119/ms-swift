#!/usr/bin/env python3
# Copyright (c) Alibaba, Inc. and its affiliates.

"""Numerical checks for DeltaNet Seq1F1B state relay.

This is intentionally a small CUDA script rather than a pytest-only test: FLA
Triton kernels are GPU-only, and the check is most useful on the same worker
image used by the Seq1F1B experiments.
"""

import argparse

import torch

from swift.megatron.deltanet.attention import DeltaNetChunkFunc, ShortConvChunkFunc


def _clone_leaf(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone().requires_grad_(True)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> None:
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected.float().abs().clamp_min(1e-6)).max().item()
    print(f'{name}: max_abs={max_abs:.4e}, max_rel={max_rel:.4e}')
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        raise AssertionError(f'{name} mismatch: max_abs={max_abs:.4e}, max_rel={max_rel:.4e}')


def check_delta_rule(args) -> None:
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    dtype = getattr(torch, args.dtype)
    scale = args.head_dim**-0.5

    q0 = torch.randn(args.batch, args.seq_len, args.heads, args.head_dim, device=device, dtype=dtype) * 0.2
    k0 = torch.randn_like(q0) * 0.2
    v0 = torch.randn_like(q0) * 0.2
    beta0 = torch.sigmoid(torch.randn(args.batch, args.seq_len, args.heads, device=device, dtype=dtype))
    dout = torch.randn_like(q0) * 0.2

    if args.qk_norms == 'l2':
        qk_norm_values = (True,)
    elif args.qk_norms == 'none':
        qk_norm_values = (False,)
    else:
        qk_norm_values = (False, True)

    for qk_norm in qk_norm_values:
        q_full, k_full, v_full, beta_full = map(_clone_leaf, (q0, k0, v0, beta0))
        out_full = DeltaNetChunkFunc.apply(
            q_full,
            k_full,
            v_full,
            beta_full,
            scale,
            {},
            qk_norm,
            args.use_ho_pipeline,
        )
        out_full.backward(dout)
        grads_full = [x.grad.detach().clone() for x in (q_full, k_full, v_full, beta_full)]

        q_split, k_split, v_split, beta_split = map(_clone_leaf, (q0, k0, v0, beta0))
        state_cache = {}
        outs = []
        for start in range(0, args.seq_len, args.chunk_len):
            end = start + args.chunk_len
            outs.append(
                DeltaNetChunkFunc.apply(
                    q_split[:, start:end],
                    k_split[:, start:end],
                    v_split[:, start:end],
                    beta_split[:, start:end],
                    scale,
                    state_cache,
                    qk_norm,
                    args.use_ho_pipeline,
                )
            )
        out_split = torch.cat(outs, dim=1)
        dout_chunks = [dout[:, start:start + args.chunk_len].contiguous()
                       for start in range(0, args.seq_len, args.chunk_len)]
        for idx in reversed(range(len(outs))):
            outs[idx].backward(dout_chunks[idx], retain_graph=idx > 0)
        grads_split = [x.grad.detach().clone() for x in (q_split, k_split, v_split, beta_split)]

        tag = f'delta_rule[qk_norm={qk_norm}]'
        _assert_close(f'{tag}/out', out_split, out_full, args.atol, args.rtol)
        for name, actual, expected in zip(('dq', 'dk', 'dv', 'dbeta'), grads_split, grads_full):
            _assert_close(f'{tag}/{name}', actual, expected, args.grad_atol, args.grad_rtol)


def check_short_conv(args) -> None:
    torch.manual_seed(args.seed + 1)
    device = torch.device('cuda')
    dtype = getattr(torch, args.dtype)

    x0 = torch.randn(args.batch, args.seq_len, args.conv_dim, device=device, dtype=dtype) * 0.2
    weight0 = torch.randn(args.conv_dim, 1, args.conv_size, device=device, dtype=dtype) * 0.2
    bias0 = torch.randn(args.conv_dim, device=device, dtype=dtype) * 0.02
    dout = torch.randn_like(x0) * 0.2

    for activation in (None, 'silu'):
        x_full, weight_full, bias_full = map(_clone_leaf, (x0, weight0, bias0))
        out_full = ShortConvChunkFunc.apply(x_full, weight_full, bias_full, {}, {}, 'x', activation)
        (out_full.float() * dout.float()).sum().backward()
        grads_full = [x.grad.detach().clone() for x in (x_full, weight_full, bias_full)]

        x_split, weight_split, bias_split = map(_clone_leaf, (x0, weight0, bias0))
        cache_dict = {}
        grad_dict = {}
        outs = []
        for start in range(0, args.seq_len, args.chunk_len):
            end = start + args.chunk_len
            outs.append(
                ShortConvChunkFunc.apply(
                    x_split[:, start:end],
                    weight_split,
                    bias_split,
                    cache_dict,
                    grad_dict,
                    'x',
                    activation,
                )
            )
        out_split = torch.cat(outs, dim=1)
        dout_chunks = [dout[:, start:start + args.chunk_len].contiguous()
                       for start in range(0, args.seq_len, args.chunk_len)]
        for idx in reversed(range(len(outs))):
            outs[idx].backward(dout_chunks[idx], retain_graph=idx > 0)
        grads_split = [x.grad.detach().clone() for x in (x_split, weight_split, bias_split)]

        tag = f'short_conv[activation={activation}]'
        _assert_close(f'{tag}/out', out_split, out_full, args.atol, args.rtol)
        for name, actual, expected in zip(('dx', 'dw', 'db'), grads_split, grads_full):
            _assert_close(f'{tag}/{name}', actual, expected, args.grad_atol, args.grad_rtol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--chunk-len', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--head-dim', type=int, default=32)
    parser.add_argument('--conv-dim', type=int, default=128)
    parser.add_argument('--conv-size', type=int, default=4)
    parser.add_argument('--use-ho-pipeline', action='store_true')
    parser.add_argument('--qk-norms', choices=['l2', 'none', 'both'], default='l2')
    parser.add_argument('--atol', type=float, default=2e-2)
    parser.add_argument('--rtol', type=float, default=2e-2)
    parser.add_argument('--grad-atol', type=float, default=4e-2)
    parser.add_argument('--grad-rtol', type=float, default=4e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('This test requires CUDA because FLA kernels are GPU-only.')
    if args.seq_len % args.chunk_len != 0:
        raise ValueError('--seq-len must be divisible by --chunk-len')

    check_short_conv(args)
    check_delta_rule(args)
    print('DeltaNet Seq1F1B full-vs-split checks passed.')


if __name__ == '__main__':
    main()
