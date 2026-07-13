# Copyright (c) Alibaba, Inc. and its affiliates.

"""Megatron parser extensions for DeltaNet/Seq1F1B."""


def add_deltanet_args(parser):
    # NOTE: Swift's MegatronArguments passes booleans by *omitting* the flag
    # when the value is False (see `MegatronArguments._args_to_argv`). All
    # store_true flags below therefore must default to False, otherwise
    # `--deltanet_use_short_conv false` on the Swift CLI would silently stay
    # True on the Megatron side. The user-facing defaults (short conv / beta /
    # output gate enabled) live in the MegatronArguments dataclass.
    group = parser.add_argument_group(title='DeltaNet')
    group.add_argument('--use-deltanet', action='store_true', default=False)
    group.add_argument('--pipe-sp-splits', type=int, default=1)
    group.add_argument(
        '--pipe-sp-strategy',
        '--pipe_sp_strategy',
        type=str,
        default='average',
        choices=['average', 'manual'],
    )
    group.add_argument('--pipe-sp-manual-splits', '--pipe_sp_manual_splits', type=str, default='')
    group.add_argument('--deltanet-mode', type=str, default='chunk', choices=['chunk', 'fused_recurrent'])
    group.add_argument('--deltanet-use-short-conv', action='store_true', default=False)
    group.add_argument('--deltanet-conv-size', type=int, default=4)
    group.add_argument('--deltanet-use-beta', action='store_true', default=False)
    group.add_argument('--deltanet-use-output-gate', action='store_true', default=False)
    group.add_argument('--deltanet-qk-activation', type=str, default='silu', choices=['silu', 'relu', 'elu', 'none'])
    group.add_argument('--deltanet-qk-norm', type=str, default='l2', choices=['l2', 'none'])
    group.add_argument('--deltanet-fused-h-o-pipeline', action='store_true', default=False)
    group.add_argument('--deltanet-allow-packed-seq', action='store_true', default=False)
    group.add_argument('--deltanet-rnn-sp1-baseline', action='store_true', default=False)
    group.add_argument('--deltanet-hybrid-attention-layers', type=str, default='')
    group.add_argument('--deltanet-hybrid-attention-period', type=int, default=0)
    group.add_argument('--deltanet-hybrid-attention-offset', type=int, default=0)

    mamba = parser.add_argument_group(title='Mamba3')
    mamba.add_argument('--use-mamba3', action='store_true', default=False)
    mamba.add_argument('--mamba3-d-state', type=int, default=128)
    mamba.add_argument('--mamba3-expand', type=int, default=2)
    mamba.add_argument('--mamba3-head-dim', type=int, default=64)
    mamba.add_argument('--mamba3-ngroups', type=int, default=1)
    mamba.add_argument('--mamba3-rope-fraction', type=float, default=0.5, choices=[0.5, 1.0])
    mamba.add_argument('--mamba3-chunk-size', type=int, default=64)
    mamba.add_argument('--mamba3-dt-min', type=float, default=0.001)
    mamba.add_argument('--mamba3-dt-max', type=float, default=0.1)
    mamba.add_argument('--mamba3-dt-init-floor', type=float, default=1e-4)
    mamba.add_argument('--mamba3-a-floor', type=float, default=1e-4)
    mamba.add_argument('--mamba3-outproj-norm', action='store_true', default=False)
    mamba.add_argument('--mamba3-is-mimo', action='store_true', default=False)
    mamba.add_argument('--mamba3-mimo-rank', type=int, default=4)
    mamba.add_argument('--mamba3-fuse-pregate-headwise-norm', action='store_true', default=False)
    return parser
