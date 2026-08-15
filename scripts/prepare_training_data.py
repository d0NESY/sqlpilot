"""一键生成 S1-S4、内部验证集和训练闸门数据。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "spider_data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "datasets"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "data" / "processed" / "reports"
DEFAULT_VALUE_CACHE = PROJECT_ROOT / "data" / "cache" / "schema_values"
DEFAULT_EXCLUSIONS = (
    PROJECT_ROOT / "configs" / "data" / "known_invalid_gold.json"
)
sys.path.insert(0, str(SRC_ROOT))

from sqlpilot.data_validation import (  # noqa: E402
    load_known_invalid_gold,
    run_spider_data_check,
    write_invalid_jsonl,
    write_json,
)
from sqlpilot.dataset_preparation import (  # noqa: E402
    load_indexed_samples,
    prepare_experiment_datasets,
    split_by_database,
)
from sqlpilot.spider_schema import load_spider_schema_catalog  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="准备 SQLPilot S1-S4 全部训练前数据"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
    )
    parser.add_argument(
        "--value-cache-root",
        type=Path,
        default=DEFAULT_VALUE_CACHE,
    )
    parser.add_argument(
        "--known-invalid-gold",
        type=Path,
        default=DEFAULT_EXCLUSIONS,
    )
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="已有完整通过报告时可跳过；首次运行不要使用",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    exclusions = load_known_invalid_gold(
        args.known_invalid_gold.resolve()
    )

    if not args.skip_data_check:
        check = run_spider_data_check(
            data_root=data_root,
            execute_gold=True,
            known_invalid_gold=exclusions,
        )
        report_root = args.report_root.resolve()
        write_json(report_root / "data_check_summary.json", check.summary)
        write_invalid_jsonl(
            report_root / "invalid_samples.jsonl",
            check.invalid_samples,
        )
        write_invalid_jsonl(
            report_root / "schema_warnings.jsonl",
            check.warnings,
        )
        write_json(
            report_root / "database_manifest.json",
            check.database_manifest,
        )
        if check.summary["status"] != "passed":
            raise SystemExit(
                "数据完整性检查未通过，已停止生成训练数据"
            )

    train_samples, excluded = load_indexed_samples(
        {
            "train_spider": data_root / "train_spider.json",
            "train_others": data_root / "train_others.json",
        },
        exclusions=exclusions,
    )
    dev_samples, _ = load_indexed_samples(
        {"dev": data_root / "dev.json"},
    )
    train, validation, validation_databases = split_by_database(
        train_samples,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    catalog = load_spider_schema_catalog(data_root / "tables.json")
    manifest = prepare_experiment_datasets(
        data_root=data_root,
        output_root=args.output_root.resolve(),
        value_cache_root=args.value_cache_root.resolve(),
        schema_catalog=catalog,
        train_samples=train,
        validation_samples=validation,
        dev_samples=dev_samples,
        validation_database_ids=validation_databases,
        excluded_gold=excluded,
        split_seed=args.seed,
        validation_ratio=args.validation_ratio,
    )
    print(
        json.dumps(
            {
                "train_record_count": manifest["train_record_count"],
                "validation_record_count": manifest[
                    "validation_record_count"
                ],
                "official_dev_record_count": manifest[
                    "official_dev_record_count"
                ],
                "validation_database_count": manifest[
                    "validation_database_count"
                ],
                "variants": list(manifest["variants"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        "数据清单："
        f"{(args.output_root.resolve() / 'dataset_manifest.json')}"
    )


if __name__ == "__main__":
    main()
