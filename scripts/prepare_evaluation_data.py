"""从现有 Spider 数据生成官方 dev Gold 和顺序清单。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sqlpilot.spider_evaluation import prepare_spider_dev_evaluation  # noqa: E402


def main() -> None:
    dataset_root = PROJECT_ROOT / "data" / "processed" / "datasets"
    result = prepare_spider_dev_evaluation(
        raw_dev_path=PROJECT_ROOT / "data" / "raw" / "spider_data" / "dev.json",
        variant_paths={
            "s1_ddl": dataset_root
            / "s1_ddl"
            / "official_dev_evaluation_only.jsonl",
            "s2_structured": dataset_root
            / "s2_structured"
            / "official_dev_evaluation_only.jsonl",
            "s3_values": dataset_root
            / "s3_values"
            / "official_dev_evaluation_only.jsonl",
            "s4_schema_link": dataset_root
            / "s4_schema_link"
            / "official_dev_evaluation_only.jsonl",
        },
        output_root=PROJECT_ROOT / "data" / "evaluation" / "spider_dev",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
