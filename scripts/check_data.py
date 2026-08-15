"""执行 Spider 开训前完整性检查并生成报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "spider_data"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "data" / "processed" / "reports"
sys.path.insert(0, str(SRC_ROOT))

from sqlpilot.data_validation import (  # noqa: E402
    load_known_invalid_gold,
    run_spider_data_check,
    write_invalid_jsonl,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 Spider 数据、Schema、隔离性、哈希和 Gold SQL"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--known-invalid-gold",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "data"
        / "known_invalid_gold.json",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
    )
    parser.add_argument(
        "--skip-gold-execution",
        action="store_true",
        help="只做结构检查；正式开训前不应跳过",
    )
    parser.add_argument(
        "--query-timeout",
        type=float,
        default=5.0,
        help="每条 Gold SQL 最长检查秒数，默认 5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.query_timeout <= 0:
        raise SystemExit("--query-timeout 必须大于 0")

    result = run_spider_data_check(
        data_root=args.data_root.resolve(),
        execute_gold=not args.skip_gold_execution,
        query_timeout_seconds=args.query_timeout,
        known_invalid_gold=load_known_invalid_gold(
            args.known_invalid_gold.resolve()
        ),
    )
    report_root = args.report_root.resolve()
    write_json(
        report_root / "data_check_summary.json",
        result.summary,
    )
    write_invalid_jsonl(
        report_root / "invalid_samples.jsonl",
        result.invalid_samples,
    )
    write_invalid_jsonl(
        report_root / "schema_warnings.jsonl",
        result.warnings,
    )
    write_json(
        report_root / "database_manifest.json",
        result.database_manifest,
    )

    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    print(f"报告目录：{report_root}")
    if result.summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
