#!/usr/bin/env bash
# Swift Megatron DeltaNet + Seq1F1B PP8/TP1 experiment.
#
# Example:
#   PIPE_SP_SPLITS=4 RUN_NAME=swift_deltanet_sp4 bash examples/train/megatron/seq1f1b_swift_deltanet_pp8tp1.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export USE_DELTANET=${USE_DELTANET:-true}
export PIPE_SP_SPLITS=${PIPE_SP_SPLITS:-4}
# Keep Swift's 16K packing path for shape/performance parity with the baseline.
# Set PACKING=false only when DATASET rows are already fixed-length SEQ_LEN samples.
export PACKING=${PACKING:-true}
export DELTANET_ALLOW_PACKED_SEQ=${DELTANET_ALLOW_PACKED_SEQ:-true}
export POSITION_EMBEDDING_TYPE=${POSITION_EMBEDDING_TYPE:-none}
export USE_FLASH_ATTN=${USE_FLASH_ATTN:-false}
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-unfused}
# `local` keeps the input LayerNorm as a separate module, so the DeltaNet beta
# projection sees post-LN hidden states exactly like the Megatron reference,
# and the strided qkvg ColumnParallelLinear also works for TP>1. With
# `transformer_engine`, the LN is fused into linear_qkvg and beta would be
# computed from pre-LN hidden states (a different parameterization).
export TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-local}
export RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-none}
export RUN_NAME=${RUN_NAME:-swift_deltanet_m2p7b_seq${SEQ_LEN:-16384}_gbs${GLOBAL_BATCH:-16}_pp${PP_SIZE:-8}tp${TP_SIZE:-1}_sp${PIPE_SP_SPLITS}}

bash "${SCRIPT_DIR}/seq1f1b_swift_pp8tp1.sh"
