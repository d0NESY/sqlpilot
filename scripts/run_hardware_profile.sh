#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${1:?用法: run_hardware_profile.sh <v100|modern> <gpu_id>}"
GPU_ID="${2:?用法: run_hardware_profile.sh <v100|modern> <gpu_id>}"
case "$PROFILE" in
  v100)
    EXPECTED_TORCH_SUFFIX="+cu126"
    ;;
  modern)
    EXPECTED_TORCH_SUFFIX="+cu128"
    ;;
  *)
    echo "未知硬件 profile：$PROFILE（只支持 v100 或 modern）" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_ROOT="configs/$PROFILE"
LOG_ROOT="logs/$PROFILE"
CHECKPOINT_ROOT="checkpoints/$PROFILE"
RESULT_ROOT="results/evaluation/$PROFILE"
mkdir -p "$LOG_ROOT"

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
    echo "[$(date --iso-8601=seconds)] $PROFILE 全部五组实验完成"
  else
    echo "[$(date --iso-8601=seconds)] $PROFILE 流程失败，exit=$status；后续实验未启动" >&2
  fi
}
trap on_exit EXIT

run_logged() {
  local name="$1"
  shift
  echo "[$(date --iso-8601=seconds)] START $name"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
  echo "[$(date --iso-8601=seconds)] DONE  $name"
}

require_fresh_training_output() {
  local output="$1"
  if [[ -f "$output/train_metrics.json" || -f "$output/adapter_config.json" ]] ||
     compgen -G "$output/checkpoint-*" > /dev/null; then
    echo "训练目录已有模型或 checkpoint，拒绝混跑：$output" >&2
    echo "如需重做，请先把该目录移动到备份位置。" >&2
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
  require_fresh_training_output "$CHECKPOINT_ROOT/$stage/seed42"
done

python - "$PROFILE" "$EXPECTED_TORCH_SUFFIX" "$CONFIG_ROOT/experiments/s1.yaml" <<'PY'
import sys
from pathlib import Path

import torch
from sqlpilot.training.config import load_training_config

profile, expected_suffix, config_path = sys.argv[1:]
root = Path.cwd()
config = load_training_config(root / config_path, root)
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用")
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
if not torch.__version__.endswith(expected_suffix):
    raise SystemExit(
        f"PyTorch wheel 不匹配：profile={profile}, "
        f"expected=*{expected_suffix}, actual={torch.__version__}"
    )
if profile == "v100":
    if "V100" not in name.upper() or capability != (7, 0):
        raise SystemExit(f"v100 profile 选中的不是 V100：{name}, CC={capability}")
    if config.bf16 or not config.fp16 or config.tf32:
        raise SystemExit("V100 配置必须是 FP16，且关闭 BF16/TF32")
else:
    if capability[0] < 8 or not torch.cuda.is_bf16_supported():
        raise SystemExit(f"modern profile GPU 不支持 BF16/TF32：{name}, CC={capability}")
    if not config.bf16 or config.fp16 or not config.tf32:
        raise SystemExit("modern 配置必须是 BF16 + TF32，且关闭 FP16")
print(
    {
        "profile": profile,
        "gpu": name,
        "compute_capability": capability,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
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
      --config "$CONFIG_ROOT/experiments/${stage}.yaml" \
      --preflight-only
done

echo "[$(date --iso-8601=seconds)] $PROFILE 的 S1-S4 全部预检通过"

if [[ -f "$RESULT_ROOT/base_s4/predictions.jsonl" ]] &&
   [[ "$(wc -l < "$RESULT_ROOT/base_s4/predictions.jsonl")" == "1034" ]]; then
  echo "[$(date --iso-8601=seconds)] SKIP  base_s4（已有 1034 条预测）"
else
  run_logged base_s4 \
    python -u scripts/predict.py \
      --config "$CONFIG_ROOT/evaluation/base_s4.yaml" \
      --resume
fi

stages=(s1 s2 s3 s4)
checkpoint_names=(s1_ddl s2_structured s3_values s4_schema_link)
for index in "${!stages[@]}"; do
  stage="${stages[$index]}"
  checkpoint_name="${checkpoint_names[$index]}"
  run_logged "${stage}_train" \
    python -u scripts/train.py \
      --config "$CONFIG_ROOT/experiments/${stage}.yaml"
  test -f "$CHECKPOINT_ROOT/$checkpoint_name/seed42/train_metrics.json"
  run_logged "${stage}_predict" \
    python -u scripts/predict.py \
      --config "$CONFIG_ROOT/evaluation/${stage}_adapter.yaml" \
      --resume
done
