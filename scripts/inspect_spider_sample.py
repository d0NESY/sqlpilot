"""查看一条 Spider 样本，并在对应 SQLite 数据库中执行 Gold SQL。

这个脚本只使用 Python 标准库，目的是帮助初学者理解：

自然语言问题 -> db_id -> 数据库 Schema -> Gold SQL -> 查询结果
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "spider_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="查看并执行一条 Spider Gold SQL 样本"
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev"),
        default="dev",
        help="读取训练集或开发集，默认 dev",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="样本下标，从 0 开始，默认 0",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Spider 数据目录",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10,
        help="最多展示多少行查询结果，默认 10",
    )
    return parser.parse_args()


def load_samples(data_root: Path, split: str) -> list[dict[str, Any]]:
    file_name = "train_spider.json" if split == "train" else "dev.json"
    json_path = data_root / file_name

    if not json_path.exists():
        raise FileNotFoundError(f"找不到 Spider 样本文件：{json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    if not isinstance(samples, list):
        raise ValueError(f"样本文件顶层应为列表：{json_path}")
    return samples


def quote_identifier(identifier: str) -> str:
    """为 SQLite 表名添加安全的双引号转义。"""

    return '"' + identifier.replace('"', '""') + '"'


def open_read_only_database(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"找不到 SQLite 数据库：{db_path}")

    uri = db_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON;")
    return connection


def read_schema(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    table_rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
        """
    ).fetchall()

    schema: list[dict[str, Any]] = []
    for (table_name,) in table_rows:
        quoted_table = quote_identifier(table_name)

        columns = []
        for row in connection.execute(
            f"PRAGMA table_info({quoted_table});"
        ).fetchall():
            _, name, data_type, not_null, default_value, primary_key = row
            columns.append(
                {
                    "name": name,
                    "type": data_type or "UNKNOWN",
                    "not_null": bool(not_null),
                    "default": default_value,
                    "primary_key": bool(primary_key),
                }
            )

        foreign_keys = []
        for row in connection.execute(
            f"PRAGMA foreign_key_list({quoted_table});"
        ).fetchall():
            foreign_keys.append(
                {
                    "target_table": row[2],
                    "source_column": row[3],
                    "target_column": row[4],
                }
            )

        schema.append(
            {
                "table": table_name,
                "columns": columns,
                "foreign_keys": foreign_keys,
            }
        )

    return schema


def print_schema(schema: list[dict[str, Any]]) -> None:
    print("\n=== Database Schema ===")

    for table in schema:
        print(f"\nTable: {table['table']}")

        for column in table["columns"]:
            properties = [column["type"]]
            if column["primary_key"]:
                properties.append("PRIMARY KEY")
            if column["not_null"]:
                properties.append("NOT NULL")

            property_text = ", ".join(properties)
            print(f"  - {column['name']}: {property_text}")

        for foreign_key in table["foreign_keys"]:
            print(
                "  - FOREIGN KEY: "
                f"{table['table']}.{foreign_key['source_column']} -> "
                f"{foreign_key['target_table']}."
                f"{foreign_key['target_column']}"
            )


def execute_gold_sql(
    connection: sqlite3.Connection,
    sql: str,
    max_rows: int,
) -> tuple[list[str], list[tuple[Any, ...]], bool]:
    cursor = connection.execute(sql)
    columns = [description[0] for description in cursor.description or []]
    rows = cursor.fetchmany(max_rows + 1)
    truncated = len(rows) > max_rows
    return columns, rows[:max_rows], truncated


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    samples = load_samples(data_root, args.split)

    if args.index < 0 or args.index >= len(samples):
        raise IndexError(
            f"index={args.index} 超出范围；"
            f"{args.split} 集共有 {len(samples)} 条样本"
        )

    sample = samples[args.index]
    question = sample["question"]
    gold_sql = sample["query"]
    db_id = sample["db_id"]
    db_path = data_root / "database" / db_id / f"{db_id}.sqlite"

    print("=== Spider Sample ===")
    print(f"Split: {args.split}")
    print(f"Index: {args.index}")
    print(f"Total samples: {len(samples)}")
    print(f"DB ID: {db_id}")
    print(f"Question: {question}")
    print(f"Database: {db_path}")

    with closing(open_read_only_database(db_path)) as connection:
        schema = read_schema(connection)
        print_schema(schema)

        print("\n=== Gold SQL ===")
        print(gold_sql)

        columns, rows, truncated = execute_gold_sql(
            connection=connection,
            sql=gold_sql,
            max_rows=args.max_rows,
        )

    print("\n=== Gold SQL Result ===")
    print(f"Columns: {columns}")
    for row in rows:
        print(row)

    if not rows:
        print("(no rows)")
    if truncated:
        print(f"... 仅展示前 {args.max_rows} 行")


if __name__ == "__main__":
    main()
