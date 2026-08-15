"""把 Spider train/dev JSON 转换成 Prompt-Completion JSONL。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "spider_data"

# 让脚本在项目包尚未安装时也能直接运行。
sys.path.insert(0, str(SRC_ROOT))

from sqlpilot.spider_schema import load_spider_schema_catalog  # noqa: E402
from sqlpilot.training_data import (  # noqa: E402
    SpiderTrainingDataBuilder,
    load_spider_samples,
    write_jsonl,
)


SPLIT_FILES = {
    "train": ("train_spider.json", "train_others.json"),
    "dev": ("dev.json",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构造 SQLPilot Prompt-Completion 训练数据"
    )
    parser.add_argument(
        "--split",
        choices=tuple(SPLIT_FILES),
        default="train",
        help="数据划分，默认 train",
    )
    parser.add_argument(
        "--style",
        choices=("ddl", "structured", "structured_with_values"),
        default="structured",
        help="Schema 序列化格式，默认 structured",
    )
    parser.add_argument(
        "--output-mode",
        choices=("direct", "schema_link"),
        default="direct",
        help="答案格式：直接 SQL 或 Schema Link + SQL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只生成前 N 条；不填写时生成整个划分",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Spider 数据目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSONL 路径；不填写时写入 data/processed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit 不能小于 0")

    data_root = args.data_root.resolve()
    source_paths = [
        data_root / file_name for file_name in SPLIT_FILES[args.split]
    ]
    output_path = args.output
    if output_path is None:
        output_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"spider_{args.split}_{args.style}_{args.output_mode}.jsonl"
        )

    samples = [
        sample
        for source_path in source_paths
        for sample in load_spider_samples(source_path)
    ]
    schema_catalog = load_spider_schema_catalog(data_root / "tables.json")
    builder = SpiderTrainingDataBuilder(
        database_root=data_root / "database",
        schema_style=args.style,
        official_schema_catalog=schema_catalog,
        value_cache_root=PROJECT_ROOT / "data" / "cache" / "schema_values",
        output_mode=args.output_mode,
    )
    records = builder.build_records(
        samples,
        split=args.split,
        limit=args.limit,
    )
    record_count = write_jsonl(records, output_path)

    print("输入文件：")
    for source_path in source_paths:
        print(f"- {source_path}")
    print(f"输出文件：{output_path.resolve()}")
    print(f"训练记录：{record_count}")
    print(f"涉及数据库：{builder.cached_database_count}")
    print(f"Schema 格式：{args.style}")
    print(f"答案格式：{args.output_mode}")


if __name__ == "__main__":
    main()
