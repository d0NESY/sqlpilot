"""读取 YAML，执行 Token 预检或正式 QLoRA 训练。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sqlpilot.hardware import apply_training_hardware  # noqa: E402
from sqlpilot.train_qlora import run_training  # noqa: E402
from sqlpilot.training_config import load_training_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SQLPilot QLoRA 训练")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--hardware",
        choices=("v100", "4090"),
        default=None,
        help="覆盖训练精度；V100=FP16，RTX 4090=BF16+TF32",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="下载 tokenizer、检查数据和 Token 长度，但不加载模型训练",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_training_hardware(
        load_training_config(args.config, PROJECT_ROOT),
        args.hardware,
    )
    result = run_training(
        config,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
