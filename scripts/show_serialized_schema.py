"""显示一个 Spider 数据库的序列化 Schema。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "spider_data"

# 让这个教学脚本在尚未安装项目包时也能直接运行。
sys.path.insert(0, str(SRC_ROOT))

from sqlpilot.data import parse_sqlite_schema, serialize_schema  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取并序列化一个 Spider SQLite Schema"
    )
    parser.add_argument(
        "--db-id",
        default="concert_singer",
        help="Spider 数据库 ID，默认 concert_singer",
    )
    parser.add_argument(
        "--style",
        choices=("ddl", "structured"),
        default="structured",
        help="序列化格式，默认 structured",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Spider 数据目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database_path = (
        args.data_root.resolve()
        / "database"
        / args.db_id
        / f"{args.db_id}.sqlite"
    )

    schema = parse_sqlite_schema(
        db_id=args.db_id,
        db_path=database_path,
    )
    print(serialize_schema(schema, style=args.style))


if __name__ == "__main__":
    main()
