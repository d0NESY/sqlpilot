"""把 predictions.jsonl 交给固定版本官方脚本计算 EM/EX/TSA。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sqlpilot.evaluation.official import (  # noqa: E402
    evaluate_official_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spider 官方 EM/EX/TSA 评估")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--official-root",
        type=Path,
        default=PROJECT_ROOT / "tools" / "official_evaluation",
    )
    parser.add_argument(
        "--test-suite-db",
        type=Path,
        default=PROJECT_ROOT
        / "tools"
        / "official_evaluation"
        / "test_suite_databases",
    )
    parser.add_argument("--timeout-seconds", type=int, default=21_600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds 必须大于 0")
    data_root = PROJECT_ROOT / "data"
    result = evaluate_official_predictions(
        predictions_jsonl=args.predictions,
        sample_manifest_jsonl=data_root
        / "evaluation"
        / "spider_dev"
        / "samples.jsonl",
        gold_path=data_root / "evaluation" / "spider_dev" / "gold.sql",
        original_database_root=data_root / "raw" / "spider_data" / "database",
        tables_path=data_root / "raw" / "spider_data" / "tables.json",
        spider_evaluator_root=args.official_root / "spider",
        test_suite_evaluator_root=args.official_root / "test-suite-sql-eval",
        test_suite_database_root=args.test_suite_db,
        output_root=args.output_dir,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
