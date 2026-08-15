#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "未检测到已激活的 Conda 环境" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

REVISION="488639f1ff808d1d3d0ba301aef8c11461451ec5"
HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
MODEL_CACHE="$HUB_CACHE/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/$REVISION"

on_exit() {
  status=$?
  if [[ $status -eq 0 ]]; then
    echo "[$(date --iso-8601=seconds)] 全部五组实验完成"
  else
    echo "[$(date --iso-8601=seconds)] 流程失败，exit=$status；后续实验未启动" >&2
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
    echo "请先把旧目录移动到备份位置，再重新启动本脚本。" >&2
    exit 2
  fi
}

if [[ -f logs/base_s4.pid ]] && kill -0 "$(cat logs/base_s4.pid)" 2>/dev/null; then
  echo "检测到已有 base_s4 进程仍在运行，拒绝并发写预测文件。" >&2
  exit 2
fi

for required in \
  config.json \
  tokenizer.json \
  model.safetensors.index.json \
  model-00001-of-00002.safetensors \
  model-00002-of-00002.safetensors; do
  if [[ ! -f "$MODEL_CACHE/$required" ]]; then
    echo "模型缓存缺少文件：$MODEL_CACHE/$required" >&2
    exit 2
  fi
done

require_fresh_training_output checkpoints/s1_ddl/seed42
require_fresh_training_output checkpoints/s2_structured/seed42
require_fresh_training_output checkpoints/s3_values/seed42
require_fresh_training_output checkpoints/s4_schema_link/seed42

python - <<'PY'
import torch
from sqlpilot.training.config import load_training_config
from pathlib import Path

root = Path.cwd()
config = load_training_config(root / "configs/experiments/s1.yaml", root)
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用")
if config.bf16 and not torch.cuda.is_bf16_supported():
    raise SystemExit("训练配置要求 BF16，但所选 GPU 不支持")
if config.tf32 and torch.cuda.get_device_capability(0)[0] < 8:
    raise SystemExit("训练配置要求 TF32，但所选 GPU 不是 Ampere 或更新架构")
print(
    {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bf16_config": config.bf16,
        "fp16_config": config.fp16,
        "tf32_config": config.tf32,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }
)
PY

run_logged stack_validation python -u scripts/validate_training_stack.py

for stage in s1 s2 s3 s4; do
  run_logged "${stage}_preflight" \
    python -u scripts/train.py \
      --config "configs/experiments/${stage}.yaml" \
      --preflight-only
done

echo "[$(date --iso-8601=seconds)] S1-S4 全部预检通过，开始五组串行实验"

baseline_complete() {
  local output="$1"
  [[ -f "$output/predictions.jsonl" ]] || return 1
  [[ "$(wc -l < "$output/predictions.jsonl")" == "1034" ]] || return 1
  python - "$output/resolved_config.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
config = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if config.get("bf16") is False else 1)
PY
}

if baseline_complete results/evaluation/base_s4_v100_fp16; then
  echo "[$(date --iso-8601=seconds)] SKIP  base_s4_v100_fp16（已有 1034 条预测）"
elif baseline_complete results/evaluation/base_s4; then
  echo "[$(date --iso-8601=seconds)] SKIP  base_s4（已有 1034 条 FP16 预测）"
else
  run_logged base_s4 \
    python -u scripts/predict.py \
      --config configs/evaluation/base_s4.yaml \
      --resume
fi

run_logged s1_train \
  python -u scripts/train.py --config configs/experiments/s1.yaml
test -f checkpoints/s1_ddl/seed42/train_metrics.json
run_logged s1_predict \
  python -u scripts/predict.py \
    --config configs/evaluation/s1_adapter.yaml \
    --resume

run_logged s2_train \
  python -u scripts/train.py --config configs/experiments/s2.yaml
test -f checkpoints/s2_structured/seed42/train_metrics.json
run_logged s2_predict \
  python -u scripts/predict.py \
    --config configs/evaluation/s2_adapter.yaml \
    --resume

run_logged s3_train \
  python -u scripts/train.py --config configs/experiments/s3.yaml
test -f checkpoints/s3_values/seed42/train_metrics.json
run_logged s3_predict \
  python -u scripts/predict.py \
    --config configs/evaluation/s3_adapter.yaml \
    --resume

run_logged s4_train \
  python -u scripts/train.py --config configs/experiments/s4.yaml
test -f checkpoints/s4_schema_link/seed42/train_metrics.json
run_logged s4_predict \
  python -u scripts/predict.py \
    --config configs/evaluation/s4_adapter.yaml \
    --resume
