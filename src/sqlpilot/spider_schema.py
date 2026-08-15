"""读取 Spider 官方 tables.json，并与 SQLite Schema 做一致性检查。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema_parser import (
    ColumnSchema,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)


SPIDER_TYPE_TO_SQL = {
    "text": "TEXT",
    "number": "NUMBER",
    "time": "TIME",
    "boolean": "BOOLEAN",
    "others": "UNKNOWN",
}


@dataclass(frozen=True)
class SpiderSchema:
    """官方 Schema 及 Spider AST 使用的表、列编号映射。"""

    database: DatabaseSchema
    table_names: tuple[str, ...]
    column_names: tuple[tuple[int, str], ...]

    def table_name(self, table_id: int) -> str:
        if table_id < 0 or table_id >= len(self.table_names):
            raise ValueError(
                f"{self.database.db_id} 中非法的 table_id：{table_id}"
            )
        return self.table_names[table_id]

    def column_identifier(self, column_id: int) -> tuple[str, str] | None:
        if column_id < 0 or column_id >= len(self.column_names):
            raise ValueError(
                f"{self.database.db_id} 中非法的 column_id：{column_id}"
            )
        table_id, column_name = self.column_names[column_id]
        if table_id == -1:
            return None
        return self.table_name(table_id), column_name


def _required_list(
    entry: dict[str, Any],
    field: str,
    db_id: str,
) -> list[Any]:
    value = entry.get(field)
    if not isinstance(value, list):
        raise ValueError(f"{db_id} 的 {field!r} 必须是列表")
    return value


def _validate_index(
    index: Any,
    upper_bound: int,
    description: str,
) -> int:
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError(f"{description} 必须是整数")
    if index < 0 or index >= upper_bound:
        raise ValueError(
            f"{description}={index} 超出合法范围 0..{upper_bound - 1}"
        )
    return index


def parse_spider_schema_entry(entry: dict[str, Any]) -> SpiderSchema:
    """把一个 tables.json 条目转换成内部 Schema。"""

    db_id = entry.get("db_id")
    if not isinstance(db_id, str) or not db_id.strip():
        raise ValueError("tables.json 条目的 db_id 必须是非空字符串")
    db_id = db_id.strip()

    raw_tables = _required_list(entry, "table_names_original", db_id)
    if not all(isinstance(name, str) and name for name in raw_tables):
        raise ValueError(f"{db_id} 存在非法表名")
    table_names = tuple(raw_tables)
    if len({name.casefold() for name in table_names}) != len(table_names):
        raise ValueError(f"{db_id} 存在重复表名")

    raw_columns = _required_list(entry, "column_names_original", db_id)
    raw_types = _required_list(entry, "column_types", db_id)
    if len(raw_columns) != len(raw_types):
        raise ValueError(f"{db_id} 的列名数量与列类型数量不一致")

    column_names: list[tuple[int, str]] = []
    for column_id, raw_column in enumerate(raw_columns):
        if (
            not isinstance(raw_column, list)
            or len(raw_column) != 2
            or not isinstance(raw_column[0], int)
            or not isinstance(raw_column[1], str)
        ):
            raise ValueError(f"{db_id} 的 column_id={column_id} 格式非法")
        table_id, column_name = raw_column
        if table_id != -1:
            _validate_index(
                table_id,
                len(table_names),
                f"{db_id}.column[{column_id}].table_id",
            )
        elif column_name != "*":
            raise ValueError(
                f"{db_id} 只有通配符列允许使用 table_id=-1"
            )
        column_names.append((table_id, column_name))

    raw_primary_keys = _required_list(entry, "primary_keys", db_id)
    primary_key_indices = [
        _validate_index(
            index,
            len(column_names),
            f"{db_id}.primary_keys",
        )
        for index in raw_primary_keys
    ]
    if any(column_names[index][0] == -1 for index in primary_key_indices):
        raise ValueError(f"{db_id} 的主键不能指向通配符列")

    primary_key_positions: dict[int, int] = {}
    positions_by_table: dict[int, int] = {}
    for column_id in primary_key_indices:
        table_id = column_names[column_id][0]
        positions_by_table[table_id] = positions_by_table.get(table_id, 0) + 1
        primary_key_positions[column_id] = positions_by_table[table_id]

    columns_by_table: dict[int, list[ColumnSchema]] = {
        table_id: [] for table_id in range(len(table_names))
    }
    for column_id, ((table_id, column_name), raw_type) in enumerate(
        zip(column_names, raw_types)
    ):
        if table_id == -1:
            continue
        if not isinstance(raw_type, str):
            raise ValueError(f"{db_id} 的 column_id={column_id} 类型非法")
        data_type = SPIDER_TYPE_TO_SQL.get(
            raw_type.casefold(),
            raw_type.upper(),
        )
        columns_by_table[table_id].append(
            ColumnSchema(
                table=table_names[table_id],
                name=column_name,
                data_type=data_type,
                not_null=False,
                default_value=None,
                primary_key_position=primary_key_positions.get(column_id, 0),
            )
        )

    raw_foreign_keys = _required_list(entry, "foreign_keys", db_id)
    foreign_keys: list[ForeignKeySchema] = []
    for pair_id, pair in enumerate(raw_foreign_keys):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{db_id} 的 foreign_key[{pair_id}] 格式非法")
        source_id = _validate_index(
            pair[0],
            len(column_names),
            f"{db_id}.foreign_key[{pair_id}].source",
        )
        target_id = _validate_index(
            pair[1],
            len(column_names),
            f"{db_id}.foreign_key[{pair_id}].target",
        )
        source = column_names[source_id]
        target = column_names[target_id]
        if source[0] == -1 or target[0] == -1:
            raise ValueError(f"{db_id} 的外键不能指向通配符列")
        foreign_keys.append(
            ForeignKeySchema(
                source_table=table_names[source[0]],
                source_column=source[1],
                target_table=table_names[target[0]],
                target_column=target[1],
            )
        )

    database = DatabaseSchema(
        db_id=db_id,
        tables=tuple(
            TableSchema(
                name=table_name,
                columns=tuple(columns_by_table[table_id]),
            )
            for table_id, table_name in enumerate(table_names)
            if not table_name.casefold().startswith("sqlite_")
        ),
        foreign_keys=tuple(
            key
            for key in foreign_keys
            if not key.source_table.casefold().startswith("sqlite_")
            and not key.target_table.casefold().startswith("sqlite_")
        ),
    )
    return SpiderSchema(
        database=database,
        table_names=table_names,
        column_names=tuple(column_names),
    )


def load_spider_schema_catalog(
    tables_json_path: str | Path,
) -> dict[str, SpiderSchema]:
    """读取整个 tables.json，并按 db_id 建立索引。"""

    path = Path(tables_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 tables.json：{path}")
    with path.open("r", encoding="utf-8") as file:
        entries = json.load(file)
    if not isinstance(entries, list):
        raise ValueError("tables.json 最外层必须是列表")

    catalog: dict[str, SpiderSchema] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("tables.json 每个条目都必须是对象")
        schema = parse_spider_schema_entry(entry)
        db_id = schema.database.db_id
        if db_id in catalog:
            raise ValueError(f"tables.json 存在重复 db_id：{db_id}")
        catalog[db_id] = schema
    return catalog


def _column_key(table: str, column: str) -> tuple[str, str]:
    return table.casefold(), column.casefold()


def compare_spider_and_sqlite_schema(
    official: DatabaseSchema,
    sqlite_schema: DatabaseSchema,
) -> list[str]:
    """比较表、列、主键和外键；返回所有不一致说明。"""

    errors: list[str] = []
    official_tables = {
        table.name.casefold(): table for table in official.tables
    }
    sqlite_tables = {
        table.name.casefold(): table for table in sqlite_schema.tables
    }

    missing_tables = sorted(official_tables.keys() - sqlite_tables.keys())
    extra_tables = sorted(sqlite_tables.keys() - official_tables.keys())
    if missing_tables:
        errors.append(f"SQLite 缺少表：{missing_tables}")
    if extra_tables:
        errors.append(f"SQLite 存在额外表：{extra_tables}")

    for table_key in sorted(official_tables.keys() & sqlite_tables.keys()):
        official_table = official_tables[table_key]
        sqlite_table = sqlite_tables[table_key]
        expected_columns = {
            column.name.casefold() for column in official_table.columns
        }
        actual_columns = {
            column.name.casefold() for column in sqlite_table.columns
        }
        missing_columns = sorted(expected_columns - actual_columns)
        extra_columns = sorted(actual_columns - expected_columns)
        if missing_columns:
            errors.append(
                f"表 {official_table.name} 的 SQLite 列缺失：{missing_columns}"
            )
        if extra_columns:
            errors.append(
                f"表 {official_table.name} 的 SQLite 额外列：{extra_columns}"
            )

    expected_primary_keys = {
        _column_key(table.name, column.name)
        for table in official.tables
        for column in table.columns
        if column.is_primary_key
    }
    actual_primary_keys = {
        _column_key(table.name, column.name)
        for table in sqlite_schema.tables
        for column in table.columns
        if column.is_primary_key
    }
    if expected_primary_keys != actual_primary_keys:
        errors.append(
            "主键不一致："
            f"official={sorted(expected_primary_keys)}, "
            f"sqlite={sorted(actual_primary_keys)}"
        )

    def foreign_key_set(
        schema: DatabaseSchema,
    ) -> set[tuple[str, str, str, str]]:
        return {
            (
                key.source_table.casefold(),
                key.source_column.casefold(),
                key.target_table.casefold(),
                key.target_column.casefold(),
            )
            for key in schema.foreign_keys
        }

    expected_foreign_keys = foreign_key_set(official)
    actual_foreign_keys = foreign_key_set(sqlite_schema)
    if expected_foreign_keys != actual_foreign_keys:
        errors.append(
            "外键不一致："
            f"official={sorted(expected_foreign_keys)}, "
            f"sqlite={sorted(actual_foreign_keys)}"
        )
    return errors
