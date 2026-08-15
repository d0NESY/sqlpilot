"""执行固定配置的零样本 baseline 或 QLoRA Adapter 推理。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sqlpilot.evaluation_config import load_evaluation_config  # noqa: E402
from sqlpilot.hardware import apply_evaluation_hardware  # noqa: E402
from sqlpilot.inference import run_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SQLPilot baseline/Adapter 确定性批量推理"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--hardware",
        choices=("v100", "4090"),
        default=None,
        help="覆盖推理精度；V100 使用 FP16，RTX 4090 使用 BF16",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从当前配置输出目录中的严格预测前缀继续",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只生成前 N 条用于冒烟；之后可去掉并加 --resume 续跑全部",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_evaluation_hardware(
        load_evaluation_config(args.config, PROJECT_ROOT),
        args.hardware,
    )
    result = run_predictions(config, resume=args.resume, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
