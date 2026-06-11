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
#   DATASET=/path/to/fineweb-edu-sample-10BT.json bash examples/train/megatron/seq1f1b_swift_pp8tp1.sh
#   DATASET=/path/to/fineweb-edu-sample-10BT.jsonl bash examples/train/megatron/seq1f1b_swift_pp8tp1.sh
#   DATASET=/mnt/hdfs/.../fineweb-edu-sample-10BT/sample/10BT/000_00000.parquet bash examples/train/megatron/seq1f1b_swift_pp8tp1.sh
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

# Swift consumes HF/ModelScope/local json/jsonl/txt/parquet datasets via --dataset.
# The Megatron .bin/.idx DATA_PREFIX used by stateflow is not accepted here.
discover_local_dataset() {
  local candidate
  for candidate in \
    "${DATA_ROOT}/fineweb-edu-sample-10BT.json" \
    "${DATA_ROOT}/fineweb-edu-sample-10BT.jsonl" \
    "${DATA_ROOT}/fineweb_edu_sample_10BT.json" \
    "${DATA_ROOT}/fineweb_edu_sample_10BT.jsonl" \
    "${DATA_ROOT}/fineweb-edu-sample-10BT.txt"; do
    if [ -f "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  for candidate in \
    "${DATA_ROOT}/fineweb-edu-sample-10BT/sample/10BT/000_00000.parquet" \
    "${DATA_ROOT}/fineweb_edu_sample_10BT/sample/10BT/000_00000.parquet"; do
    if [ -f "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  for candidate in \
    "${DATA_ROOT}/fineweb-edu-sample-10BT" \
    "${DATA_ROOT}/fineweb_edu_sample_10BT"; do
    if [ -d "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  candidate=$(find "${DATA_ROOT}" -maxdepth 5 -type f \
    \( -name '*.json' -o -name '*.jsonl' -o -name '*.parquet' -o -name '*.txt' \) \
    2>/dev/null | head -n 1 || true)
  if [ -n "${candidate}" ]; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  return 1
}

DATASET=${DATASET:-$(discover_local_dataset || true)}
USE_HF=${USE_HF:-false}
STREAMING=${STREAMING:-true}
ALLOW_REMOTE_DATASET=${ALLOW_REMOTE_DATASET:-false}

MODEL_DIR=${MODEL_DIR:-/opt/tiger/swift-gpt2-2p7b-config}
SAVE_ROOT=${SAVE_ROOT:-/opt/tiger/swift_runs}

SEQ_LEN=${SEQ_LEN:-16384}
MICRO_BATCH=${MICRO_BATCH:-1}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
TRAIN_ITERS=${TRAIN_ITERS:-30}
WARMUP_ITERS=${WARMUP_ITERS:-4}
LOG_INTERVAL=${LOG_INTERVAL:-1}
SUMMARY_SKIP=${SUMMARY_SKIP:-1}
EVAL_ITERS=${EVAL_ITERS:-0}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}
NO_SAVE_MODEL=${NO_SAVE_MODEL:-true}
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-none}

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

if [ $((NPROC_PER_NODE % (TP_SIZE * PP_SIZE))) -ne 0 ]; then
  echo "ERROR: NPROC_PER_NODE (${NPROC_PER_NODE}) must be divisible by TP_SIZE*PP_SIZE ($((TP_SIZE * PP_SIZE)))." >&2
  exit 1
fi
DP_SIZE=$((NPROC_PER_NODE / (TP_SIZE * PP_SIZE)))
if [ $((GLOBAL_BATCH % (MICRO_BATCH * DP_SIZE))) -ne 0 ]; then
  echo "ERROR: GLOBAL_BATCH (${GLOBAL_BATCH}) must be divisible by MICRO_BATCH*DP_SIZE ($((MICRO_BATCH * DP_SIZE)))." >&2
  exit 1
fi
NUM_MICROBATCHES=$((GLOBAL_BATCH / (MICRO_BATCH * DP_SIZE)))
GRADIENT_ACCUMULATION_STEPS=${NUM_MICROBATCHES}

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

DATASET_PATH_FOR_CHECK="${DATASET%%#*}"
DATASET_PATH_FOR_CHECK="${DATASET_PATH_FOR_CHECK%%:*}"
if [ -z "${DATASET}" ]; then
  echo "ERROR: DATASET is empty and no local json/jsonl/parquet/txt was found under DATA_ROOT." >&2
  echo "  DATA_ROOT=${DATA_ROOT}" >&2
  echo "Find a local Swift-readable source with:" >&2
  echo "  find ${DATA_ROOT} -maxdepth 5 -type f \\( -name '*.json' -o -name '*.jsonl' -o -name '*.parquet' -o -name '*.txt' \\) | head" >&2
  echo "Then run with DATASET=/path/to/file USE_HF=false." >&2
  echo "Note: Megatron .bin/.idx files are not accepted by Swift's --dataset path." >&2
  exit 1
elif [ -e "${DATASET_PATH_FOR_CHECK}" ]; then
  :
elif [[ "${DATASET}" == /* || "${DATASET}" == ./* || "${DATASET}" == ../* ]]; then
  echo "ERROR: local DATASET path does not exist: ${DATASET_PATH_FOR_CHECK}" >&2
  echo "Set DATASET to a local json/jsonl/parquet/txt file or folder. For example:" >&2
  echo "  DATASET=${DATA_ROOT}/fineweb-edu-sample-10BT.json USE_HF=false bash $0" >&2
  echo "Note: ${DATA_ROOT}/fineweb_edu_sample_10BT_text_document.{bin,idx} is for Megatron only, not Swift." >&2
  exit 1
elif [ "${ALLOW_REMOTE_DATASET}" != "true" ]; then
  echo "ERROR: DATASET is not a local path: ${DATASET}" >&2
  echo "This GPU environment appears offline; pass a local json/jsonl/parquet/txt path instead." >&2
  echo "If you intentionally want a hub dataset, set ALLOW_REMOTE_DATASET=true and USE_HF=true/false." >&2
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
echo "  SAVE_INTERVAL=${SAVE_INTERVAL}, NO_SAVE_MODEL=${NO_SAVE_MODEL}"
echo "  GPUs=${NPROC_PER_NODE}, PP=${PP_SIZE}, TP=${TP_SIZE}, DP=${DP_SIZE}"
echo "  model: L=${NUM_LAYERS}, H=${HIDDEN_SIZE}, heads=${NUM_HEADS}, ffn=${FFN_HIDDEN_SIZE}"
echo "  seq_len=${SEQ_LEN}, micro=${MICRO_BATCH}, global=${GLOBAL_BATCH}, iters=${TRAIN_ITERS}"
echo "  num_microbatches=${NUM_MICROBATCHES}, gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
echo "  recompute_granularity=${RECOMPUTE_GRANULARITY}"
echo "  summary_skip=${SUMMARY_SKIP}"
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
  --recompute_granularity "${RECOMPUTE_GRANULARITY}" \
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
  --save "${SAVE}/meta" \
  --no_save_model "${NO_SAVE_MODEL}" \
  --add_version false \
  --no_save_optim true \
  --no_save_rng true \
  --extra_megatron_kwargs '{"init_method_std": 0.006, "initial_loss_scale": 65536}' \
  2>&1 | tee "${SAVE}/train.log"

python3 - <<PY
from pathlib import Path
import math
import re

log_path = Path("${SAVE}") / "train.log"
seq_len = int("${SEQ_LEN}")
global_batch = int("${GLOBAL_BATCH}")
summary_skip = int("${SUMMARY_SKIP}")
text = log_path.read_text(errors="replace")
lines = text.splitlines()

time_re = re.compile(r"elapsed time per iteration \\(ms\\):\\s*([0-9.]+)")
tflops_re = re.compile(r"throughput per GPU \\(TFLOP/s/GPU\\):\\s*([0-9.]+)")
mem_stage_re = re.compile(r"mem_each_stage:\\s*([0-9.,]+)")
mem_re = re.compile(
    r"\\[Rank\\s+(?P<rank>\\d+)\\].*?"
    r"max allocated:\\s*(?P<alloc>[0-9.eE+-]+).*?"
    r"max reserved:\\s*(?P<reserved>[0-9.eE+-]+)"
)

times_s = [float(m.group(1)) / 1000.0 for line in lines for m in [time_re.search(line)] if m]
toks = [global_batch * seq_len / t for t in times_s if t > 0]
tflops = [float(m.group(1)) for line in lines for m in [tflops_re.search(line)] if m]

mem_each_stage = ""
for line in lines:
    m = mem_stage_re.search(line)
    if m:
        mem_each_stage = "/".join(m.group(1).split(","))

mem_alloc_mb_by_rank = {}
mem_reserved_mb_by_rank = {}
for line in lines:
    for m in mem_re.finditer(line):
        rank = int(m.group("rank"))
        alloc = float(m.group("alloc"))
        reserved = float(m.group("reserved"))
        mem_alloc_mb_by_rank[rank] = max(mem_alloc_mb_by_rank.get(rank, 0.0), alloc)
        mem_reserved_mb_by_rank[rank] = max(mem_reserved_mb_by_rank.get(rank, 0.0), reserved)

def drop_warmup(values):
    if len(values) > summary_skip:
        return values[summary_skip:]
    return values

def mean_std(values):
    values = list(values)
    if not values:
        return None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)

def pm(values, scale=1.0):
    stats = mean_std(drop_warmup(values))
    if stats is None:
        return "NA"
    mean, std = stats
    return f"{mean * scale:.2f}\\u00b1{std * scale:.2f}"

def fmt_mem_gb(mb):
    gb = mb / 1024.0
    if gb >= 10:
        return f"{gb:.1f}"
    return f"{gb:.2f}"

summary_lines = [
    f"time:  {pm(times_s)}",
    f"toks:  {pm(toks)}",
    f"tflops:  {pm(tflops)}",
]

if mem_each_stage:
    summary_lines.append(f"mem_arr:  {mem_each_stage}")
elif mem_alloc_mb_by_rank:
    ranks = range(max(mem_alloc_mb_by_rank) + 1)
    mem_arr = "/".join(fmt_mem_gb(mem_alloc_mb_by_rank[r]) for r in ranks if r in mem_alloc_mb_by_rank)
    summary_lines.append(f"mem_arr:  {mem_arr}")
else:
    summary_lines.append("mem_arr:  NA")

summary_lines.append(f"summary_log: {log_path}")
summary = "\\n".join(summary_lines)
print(summary)

with log_path.open("a", encoding="utf-8") as f:
    f.write("\\n" + summary + "\\n")
PY
