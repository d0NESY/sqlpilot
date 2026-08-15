"""从 SQLite 数据库读取表、字段、主键和外键。"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ColumnSchema:
    """一列的 Schema 信息。"""

    table: str
    name: str
    data_type: str
    not_null: bool
    default_value: str | None
    primary_key_position: int

    @property
    def is_primary_key(self) -> bool:
        return self.primary_key_position > 0


@dataclass(frozen=True)
class ForeignKeySchema:
    """一条外键关系。"""

    source_table: str
    source_column: str
    target_table: str
    target_column: str


@dataclass(frozen=True)
class TableSchema:
    """一张表及其所有字段。"""

    name: str
    columns: tuple[ColumnSchema, ...]


@dataclass(frozen=True)
class DatabaseSchema:
    """一个数据库的完整 Schema。"""

    db_id: str
    tables: tuple[TableSchema, ...]
    foreign_keys: tuple[ForeignKeySchema, ...]


def quote_identifier(identifier: str) -> str:
    """使用 SQLite 双引号规则转义表名。"""

    return '"' + identifier.replace('"', '""') + '"'


def open_read_only_database(db_path: Path) -> sqlite3.Connection:
    """以只读方式打开 SQLite 数据库。"""

    if not db_path.is_file():
        raise FileNotFoundError(f"找不到 SQLite 数据库：{db_path}")

    uri = db_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON;")
    return connection


def parse_sqlite_schema(db_id: str, db_path: str | Path) -> DatabaseSchema:
    """从 SQLite 元数据中读取数据库 Schema。

    Args:
        db_id: Spider 中的数据库标识。
        db_path: SQLite 文件路径。

    Returns:
        只包含不可变数据对象的 DatabaseSchema。
    """

    if not db_id.strip():
        raise ValueError("db_id 不能为空")

    path = Path(db_path)
    tables: list[TableSchema] = []
    foreign_keys: list[ForeignKeySchema] = []

    with closing(open_read_only_database(path)) as connection:
        table_names = [
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name;
                """
            ).fetchall()
        ]

        for table_name in table_names:
            quoted_table = quote_identifier(table_name)
            column_rows = connection.execute(
                f"PRAGMA table_info({quoted_table});"
            ).fetchall()

            columns = tuple(
                ColumnSchema(
                    table=table_name,
                    name=row[1],
                    data_type=row[2] or "UNKNOWN",
                    not_null=bool(row[3]),
                    default_value=row[4],
                    primary_key_position=int(row[5]),
                )
                for row in column_rows
            )
            tables.append(TableSchema(name=table_name, columns=columns))

            foreign_key_rows = connection.execute(
                f"PRAGMA foreign_key_list({quoted_table});"
            ).fetchall()
            foreign_keys.extend(
                ForeignKeySchema(
                    source_table=table_name,
                    source_column=row[3],
                    target_table=row[2],
                    target_column=row[4],
                )
                for row in foreign_key_rows
            )

    return DatabaseSchema(
        db_id=db_id,
        tables=tuple(tables),
        foreign_keys=tuple(foreign_keys),
    )
