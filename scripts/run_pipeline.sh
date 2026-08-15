#!/usr/bin/env bash
set -Eeuo pipefail

HARDWARE="${1:-v100}"
GPU_ID="${2:-0}"
case "$HARDWARE" in
  v100)
    EXPECTED_TORCH_SUFFIX="+cu126"
    ;;
  4090)
    EXPECTED_TORCH_SUFFIX="+cu128"
    ;;
  *)
    echo "用法: bash scripts/run_pipeline.sh <v100|4090> [gpu_id]" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "未检测到已激活的 Conda 环境" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

OFFLINE="${SQLPILOT_OFFLINE:-1}"
if [[ "$OFFLINE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
else
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
fi

REVISION="488639f1ff808d1d3d0ba301aef8c11461451ec5"
HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
MODEL_CACHE="$HUB_CACHE/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/$REVISION"

on_exit() {
  status=$?
  if [[ $status -eq 0 ]]; then
    echo "[$(date --iso-8601=seconds)] 全部实验完成"
  else
    echo "[$(date --iso-8601=seconds)] 流程失败，exit=$status" >&2
  fi
}
trap on_exit EXIT

run_logged() {
  local name="$1"
  shift
  echo "[$(date --iso-8601=seconds)] START $name"
  "$@" 2>&1 | tee "logs/${name}.log"
  echo "[$(date --iso-8601=seconds)] DONE  $name"
}

require_fresh_training_output() {
  local output="$1"
  if [[ -f "$output/train_metrics.json" || -f "$output/adapter_config.json" ]] ||
     compgen -G "$output/checkpoint-*" > /dev/null; then
    echo "训练目录已有模型或 checkpoint，拒绝混跑：$output" >&2
    exit 2
  fi
}

if [[ "$OFFLINE" == "1" ]]; then
  for required in \
    config.json \
    tokenizer.json \
    model.safetensors.index.json \
    model-00001-of-00002.safetensors \
    model-00002-of-00002.safetensors; do
    if [[ ! -f "$MODEL_CACHE/$required" ]]; then
      echo "离线模型缓存缺少：$MODEL_CACHE/$required" >&2
      exit 2
    fi
  done
fi

for stage in s1_ddl s2_structured s3_values s4_schema_link; do
  require_fresh_training_output "checkpoints/$stage/seed42"
done

python - "$HARDWARE" "$EXPECTED_TORCH_SUFFIX" <<'PY'
import sys
from pathlib import Path

import torch

from sqlpilot.hardware import apply_training_hardware
from sqlpilot.training_config import load_training_config

hardware, expected_suffix = sys.argv[1:]
root = Path.cwd()
config = apply_training_hardware(
    load_training_config(root / "configs/experiments/s1.yaml", root),
    hardware,
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用")
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
if not torch.__version__.endswith(expected_suffix):
    raise SystemExit(
        f"PyTorch wheel 不匹配：expected=*{expected_suffix}, "
        f"actual={torch.__version__}"
    )
if hardware == "v100":
    if "V100" not in name.upper() or capability != (7, 0):
        raise SystemExit(f"选择的 GPU 不是 V100：{name}, CC={capability}")
else:
    if "4090" not in name.upper() or not torch.cuda.is_bf16_supported():
        raise SystemExit(f"选择的 GPU 不是支持 BF16 的 RTX 4090：{name}")
print(
    {
        "hardware": hardware,
        "gpu": name,
        "compute_capability": capability,
        "torch": torch.__version__,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "tf32": config.tf32,
    }
)
PY

run_logged stack_validation python -u scripts/validate_training_stack.py

for stage in s1 s2 s3 s4; do
  run_logged "${stage}_preflight" \
    python -u scripts/train.py \
      --config "configs/experiments/${stage}.yaml" \
      --hardware "$HARDWARE" \
      --preflight-only
done

if [[ -f results/evaluation/base_s4/predictions.jsonl ]] &&
   [[ "$(wc -l < results/evaluation/base_s4/predictions.jsonl)" == "1034" ]]; then
  echo "[$(date --iso-8601=seconds)] SKIP  base_s4（已有 1034 条预测）"
else
  run_logged base_s4 \
    python -u scripts/predict.py \
      --config configs/evaluation/base_s4.yaml \
      --hardware "$HARDWARE" \
      --resume
fi

stages=(s1 s2 s3 s4)
checkpoint_names=(s1_ddl s2_structured s3_values s4_schema_link)
for index in "${!stages[@]}"; do
  stage="${stages[$index]}"
  checkpoint_name="${checkpoint_names[$index]}"
  run_logged "${stage}_train" \
    python -u scripts/train.py \
      --config "configs/experiments/${stage}.yaml" \
      --hardware "$HARDWARE"
  test -f "checkpoints/$checkpoint_name/seed42/train_metrics.json"
  run_logged "${stage}_predict" \
    python -u scripts/predict.py \
      --config "configs/evaluation/${stage}_adapter.yaml" \
      --hardware "$HARDWARE" \
      --resume
done
