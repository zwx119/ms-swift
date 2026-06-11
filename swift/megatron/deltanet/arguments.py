# Copyright (c) Alibaba, Inc. and its affiliates.

"""Megatron parser extensions for DeltaNet/Seq1F1B."""


def add_deltanet_args(parser):
    group = parser.add_argument_group(title='DeltaNet')
    group.add_argument('--use-deltanet', action='store_true', default=False)
    group.add_argument('--pipe-sp-splits', type=int, default=1)
    group.add_argument('--deltanet-mode', type=str, default='chunk', choices=['chunk', 'fused_recurrent'])
    group.add_argument('--deltanet-use-short-conv', action='store_true', default=True)
    group.add_argument('--deltanet-conv-size', type=int, default=4)
    group.add_argument('--deltanet-use-beta', action='store_true', default=True)
    group.add_argument('--deltanet-use-output-gate', action='store_true', default=True)
    group.add_argument('--deltanet-qk-activation', type=str, default='silu', choices=['silu', 'relu', 'elu', 'none'])
    group.add_argument('--deltanet-qk-norm', type=str, default='l2', choices=['l2', 'none'])
    group.add_argument('--deltanet-fused-h-o-pipeline', action='store_true', default=False)
    group.add_argument('--deltanet-allow-packed-seq', action='store_true', default=False)
    return parser

