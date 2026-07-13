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
export POSITION_EMBEDDING_TYPE=${POSITION_EMBEDDING_TYPE:-rope}
export USE_FLASH_ATTN=${USE_FLASH_ATTN:-true}
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-flash}
# Keep Swift/Megatron baseline optimizations on by default: TE layer specs,
# flash attention backend flags, and RoPE config. DeltaNet replaces the
# self-attention module; TE still provides fused LN+linear projections and
# TE MLP/projection modules where the layer spec supports them.
#
# Note: DeltaNet itself ignores rotary_pos_emb, matching the stateflow
# DeltaNet path; POSITION_EMBEDDING_TYPE=rope keeps the surrounding Swift
# model config aligned with the Swift baseline.
export TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-transformer_engine}
export RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-none}
export RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-}
export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-}
export RUN_NAME=${RUN_NAME:-swift_deltanet_m2p7b_seq${SEQ_LEN:-16384}_gbs${GLOBAL_BATCH:-16}_pp${PP_SIZE:-8}tp${TP_SIZE:-1}_sp${PIPE_SP_SPLITS}}

bash "${SCRIPT_DIR}/seq1f1b_swift_pp8tp1.sh"
