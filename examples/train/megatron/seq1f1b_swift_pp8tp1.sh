#!/usr/bin/env bash
# Swift Megatron baseline for the Seq1F1B DeltaNet main-result setting.
#
# Default setting:
#   - 8 GPUs, PP=8, TP=1
#   - 32 layers, hidden 2560, 32 heads
#   - seq_len 16K, global batch 16
#   - GPT-2 BPE tokenizer files from the stateflow data root
#
# Override examples:
#   DATASET=/path/to/fineweb_swift.jsonl bash examples/train/megatron/seq1f1b_swift_pp8tp1.sh
#   SEQ_LEN=32768 TRAIN_ITERS=30 bash examples/train/megatron/seq1f1b_swift_pp8tp1.sh

set -euo pipefail

MS_SWIFT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

# Keep official Megatron-LM separate from /opt/tiger/stateflow.
export MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-/opt/tiger/Megatron-LM-core_r0.12.0}
export PYTHONPATH="${MS_SWIFT_ROOT}:${MEGATRON_LM_PATH}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

DATA_ROOT=${DATA_ROOT:-/mnt/hdfs/ad_content.db/reasoning/minicpm/stateflow_data/data}
VOCAB=${VOCAB:-${DATA_ROOT}/gpt2/gpt2-vocab.json}
MERGE=${MERGE:-${DATA_ROOT}/gpt2/gpt2-merges.txt}

# Swift consumes HF/ModelScope/local jsonl/txt/parquet datasets via --dataset.
# The Megatron .bin/.idx DATA_PREFIX used by stateflow is not accepted here.
DATASET=${DATASET:-HuggingFaceFW/fineweb-edu:sample-10BT}
USE_HF=${USE_HF:-true}
STREAMING=${STREAMING:-true}

MODEL_DIR=${MODEL_DIR:-/opt/tiger/swift-gpt2-2p7b-config}
SAVE_ROOT=${SAVE_ROOT:-/opt/tiger/swift_runs}

SEQ_LEN=${SEQ_LEN:-16384}
MICRO_BATCH=${MICRO_BATCH:-1}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
TRAIN_ITERS=${TRAIN_ITERS:-30}
WARMUP_ITERS=${WARMUP_ITERS:-4}
LOG_INTERVAL=${LOG_INTERVAL:-1}
EVAL_ITERS=${EVAL_ITERS:-0}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-8}
NUM_LAYERS=${NUM_LAYERS:-32}
HIDDEN_SIZE=${HIDDEN_SIZE:-2560}
NUM_HEADS=${NUM_HEADS:-32}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-10240}
PADDED_VOCAB_SIZE=${PADDED_VOCAB_SIZE:-50304}

LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-3e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
CLIP_GRAD=${CLIP_GRAD:-1.0}

RUN_NAME=${RUN_NAME:-swift_m2p7b_seq${SEQ_LEN}_gbs${GLOBAL_BATCH}_pp${PP_SIZE}tp${TP_SIZE}}
SAVE=${SAVE:-${SAVE_ROOT}/${RUN_NAME}}
mkdir -p "${SAVE}" "${MODEL_DIR}"

if [ ! -d "${MEGATRON_LM_PATH}/megatron" ]; then
  echo "ERROR: MEGATRON_LM_PATH is not an official Megatron-LM checkout: ${MEGATRON_LM_PATH}" >&2
  echo "Expected the core_r0.12.0 checkout, for example /opt/tiger/Megatron-LM-core_r0.12.0" >&2
  exit 1
fi

if [ ! -f "${VOCAB}" ] || [ ! -f "${MERGE}" ]; then
  echo "ERROR: GPT-2 vocab/merges missing." >&2
  echo "  VOCAB=${VOCAB}" >&2
  echo "  MERGE=${MERGE}" >&2
  exit 1
fi

cp "${VOCAB}" "${MODEL_DIR}/vocab.json"
cp "${MERGE}" "${MODEL_DIR}/merges.txt"

python3 - <<PY
import json
from pathlib import Path

model_dir = Path("${MODEL_DIR}")
seq_len = int("${SEQ_LEN}")
padded_vocab_size = int("${PADDED_VOCAB_SIZE}")
hidden_size = int("${HIDDEN_SIZE}")
num_layers = int("${NUM_LAYERS}")
num_heads = int("${NUM_HEADS}")
ffn_hidden_size = int("${FFN_HIDDEN_SIZE}")

config = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": padded_vocab_size,
    "hidden_size": hidden_size,
    "intermediate_size": ffn_hidden_size,
    "num_hidden_layers": num_layers,
    "num_attention_heads": num_heads,
    "num_key_value_heads": num_heads,
    "max_position_embeddings": seq_len,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000,
    "tie_word_embeddings": False,
    "hidden_act": "gelu",
    "attention_bias": True,
    "mlp_bias": True,
    "attention_dropout": 0.0,
    "torch_dtype": "bfloat16",
}
tokenizer_config = {
    "tokenizer_class": "GPT2Tokenizer",
    "model_max_length": seq_len,
    "bos_token": "<|endoftext|>",
    "eos_token": "<|endoftext|>",
    "unk_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>",
}
special_tokens = {
    "bos_token": "<|endoftext|>",
    "eos_token": "<|endoftext|>",
    "unk_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>",
}
for name, obj in {
    "config.json": config,
    "tokenizer_config.json": tokenizer_config,
    "special_tokens_map.json": special_tokens,
}.items():
    (model_dir / name).write_text(json.dumps(obj, indent=2) + "\n")
print(f"Prepared model config at {model_dir}")
PY

echo "======================================================================"
echo "Swift Megatron PP/TP baseline"
echo "  MS_SWIFT_ROOT=${MS_SWIFT_ROOT}"
echo "  MEGATRON_LM_PATH=${MEGATRON_LM_PATH}"
echo "  DATASET=${DATASET}"
echo "  USE_HF=${USE_HF}"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  SAVE=${SAVE}"
echo "  GPUs=${NPROC_PER_NODE}, PP=${PP_SIZE}, TP=${TP_SIZE}"
echo "  model: L=${NUM_LAYERS}, H=${HIDDEN_SIZE}, heads=${NUM_HEADS}, ffn=${FFN_HIDDEN_SIZE}"
echo "  seq_len=${SEQ_LEN}, micro=${MICRO_BATCH}, global=${GLOBAL_BATCH}, iters=${TRAIN_ITERS}"
echo "======================================================================"

megatron pt \
  --model "${MODEL_DIR}" \
  --model_type llama \
  --dataset "${DATASET}" \
  --use_hf "${USE_HF}" \
  --streaming "${STREAMING}" \
  --packing true \
  --tensor_model_parallel_size "${TP_SIZE}" \
  --pipeline_model_parallel_size "${PP_SIZE}" \
  --micro_batch_size "${MICRO_BATCH}" \
  --global_batch_size "${GLOBAL_BATCH}" \
  --num_layers "${NUM_LAYERS}" \
  --hidden_size "${HIDDEN_SIZE}" \
  --ffn_hidden_size "${FFN_HIDDEN_SIZE}" \
  --num_attention_heads "${NUM_HEADS}" \
  --num_query_groups "${NUM_HEADS}" \
  --padded_vocab_size "${PADDED_VOCAB_SIZE}" \
  --seq_length "${SEQ_LEN}" \
  --max_length "${SEQ_LEN}" \
  --max_position_embeddings "${SEQ_LEN}" \
  --position_embedding_type rope \
  --normalization LayerNorm \
  --swiglu false \
  --disable_bias_linear false \
  --add_qkv_bias true \
  --untie_embeddings_and_output_weights true \
  --attention_dropout 0 \
  --hidden_dropout 0 \
  --bf16 true \
  --no_initialization false \
  --use_flash_attn true \
  --attention_backend flash \
  --use_distributed_optimizer true \
  --cross_entropy_loss_fusion true \
  --recompute_granularity selective \
  --train_iters "${TRAIN_ITERS}" \
  --eval_iters "${EVAL_ITERS}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --save_interval "${SAVE_INTERVAL}" \
  --lr "${LR}" \
  --min_lr "${MIN_LR}" \
  --lr_decay_style cosine \
  --lr_warmup_iters "${WARMUP_ITERS}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --clip_grad "${CLIP_GRAD}" \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --log_interval "${LOG_INTERVAL}" \
  --log_throughput true \
  --tensorboard_dir "${SAVE}/tb" \
  --save "${SAVE}/ckpt" \
  --add_version false \
  --no_save_optim true \
  --no_save_rng true \
  --extra_megatron_kwargs '{"init_method_std": 0.006, "initial_loss_scale": 65536}' \
  2>&1 | tee "${SAVE}/train.log"

python3 - <<PY
from pathlib import Path
import re

log_path = Path("${SAVE}") / "train.log"
seq_len = int("${SEQ_LEN}")
global_batch = int("${GLOBAL_BATCH}")
pattern = re.compile(r"elapsed time per iteration \\(ms\\):\\s*([0-9.]+)")
times = [float(m.group(1)) for line in log_path.read_text(errors="replace").splitlines() for m in [pattern.search(line)] if m]
if times:
    # Skip the first logged iteration when possible.
    used = times[1:] or times
    avg_ms = sum(used) / len(used)
    toks = global_batch * seq_len / (avg_ms / 1000.0)
    print(f"SUMMARY avg_iter_ms={avg_ms:.2f} tokens_per_sec={toks:.2f} log={log_path}")
else:
    print(f"SUMMARY no iteration time found; inspect {log_path}")
PY
