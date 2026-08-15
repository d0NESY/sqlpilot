"""把 DatabaseSchema 转换成适合大模型阅读的文本。"""

from __future__ import annotations

import re
import json
from collections import defaultdict
from typing import Any

from .schema_parser import DatabaseSchema, ForeignKeySchema, TableSchema


SIMPLE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SUPPORTED_STYLES = {"ddl", "structured", "structured_with_values"}
DEFAULT_EXAMPLE_CHARACTER_BUDGET = 3000


def render_identifier(identifier: str) -> str:
    """仅在标识符需要时添加 SQL 双引号。"""

    if SIMPLE_IDENTIFIER_PATTERN.fullmatch(identifier):
        return identifier
    return '"' + identifier.replace('"', '""') + '"'


def _foreign_keys_by_source(
    schema: DatabaseSchema,
) -> dict[str, list[ForeignKeySchema]]:
    grouped: dict[str, list[ForeignKeySchema]] = defaultdict(list)
    for foreign_key in schema.foreign_keys:
        grouped[foreign_key.source_table].append(foreign_key)
    return grouped


def _serialize_table_as_ddl(
    table: TableSchema,
    foreign_keys: list[ForeignKeySchema],
) -> str:
    lines = [f"CREATE TABLE {render_identifier(table.name)} ("]
    definitions: list[str] = []

    primary_key_columns = sorted(
        (column for column in table.columns if column.is_primary_key),
        key=lambda column: column.primary_key_position,
    )
    has_composite_primary_key = len(primary_key_columns) > 1

    for column in table.columns:
        parts = [
            f"  {render_identifier(column.name)}",
            column.data_type,
        ]
        if column.is_primary_key and not has_composite_primary_key:
            parts.append("PRIMARY KEY")
        if column.not_null:
            parts.append("NOT NULL")
        definitions.append(" ".join(parts))

    if has_composite_primary_key:
        primary_key_text = ", ".join(
            render_identifier(column.name)
            for column in primary_key_columns
        )
        definitions.append(f"  PRIMARY KEY ({primary_key_text})")

    for foreign_key in foreign_keys:
        definitions.append(
            "  FOREIGN KEY "
            f"({render_identifier(foreign_key.source_column)}) "
            f"REFERENCES {render_identifier(foreign_key.target_table)} "
            f"({render_identifier(foreign_key.target_column)})"
        )

    lines.append(",\n".join(definitions))
    lines.append(");")
    return "\n".join(lines)


def serialize_as_ddl(schema: DatabaseSchema) -> str:
    """序列化成接近 CREATE TABLE 的 DDL 文本。"""

    foreign_keys = _foreign_keys_by_source(schema)
    table_blocks = [
        _serialize_table_as_ddl(
            table=table,
            foreign_keys=foreign_keys.get(table.name, []),
        )
        for table in schema.tables
    ]
    return "\n\n".join(table_blocks)


def serialize_as_structured(
    schema: DatabaseSchema,
    example_values: dict[str, dict[str, list[Any]]] | None = None,
    max_example_characters: int = DEFAULT_EXAMPLE_CHARACTER_BUDGET,
) -> str:
    """序列化成人更容易阅读的结构化文本。"""

    if max_example_characters < 0:
        raise ValueError("max_example_characters 不能小于 0")
    budgeted_values = _limit_example_values(
        schema,
        example_values or {},
        max_example_characters,
    )
    lines = [f"Database: {schema.db_id}"]

    for table in schema.tables:
        lines.extend(["", f"Table: {table.name}"])
        for column in table.columns:
            properties = [column.data_type]
            if column.is_primary_key:
                properties.append("PRIMARY KEY")
            if column.not_null:
                properties.append("NOT NULL")
            values = budgeted_values.get(table.name, {}).get(
                column.name,
                [],
            )
            if values:
                rendered_values = json.dumps(values, ensure_ascii=False)
                properties.append(f"examples={rendered_values}")
            lines.append(
                f"- {column.name}: {', '.join(properties)}"
            )

    lines.extend(["", "Foreign keys:"])
    if schema.foreign_keys:
        for foreign_key in schema.foreign_keys:
            lines.append(
                f"- {foreign_key.source_table}."
                f"{foreign_key.source_column} -> "
                f"{foreign_key.target_table}."
                f"{foreign_key.target_column}"
            )
    else:
        lines.append("- None")

    return "\n".join(lines)


def _limit_example_values(
    schema: DatabaseSchema,
    values: dict[str, dict[str, list[Any]]],
    max_characters: int,
) -> dict[str, dict[str, list[Any]]]:
    """按列轮询分配总字符预算，避免大 Schema 被示例值撑爆。"""

    columns = [
        (table.name, column.name)
        for table in schema.tables
        for column in table.columns
        if values.get(table.name, {}).get(column.name)
    ]
    selected: dict[str, dict[str, list[Any]]] = {}
    used = 0
    max_value_count = max(
        (
            len(values[table_name][column_name])
            for table_name, column_name in columns
        ),
        default=0,
    )

    for value_index in range(max_value_count):
        for table_name, column_name in columns:
            candidates = values[table_name][column_name]
            if value_index >= len(candidates):
                continue
            rendered = json.dumps(
                candidates[value_index],
                ensure_ascii=False,
            )
            current = selected.get(table_name, {}).get(column_name, [])
            additional = len(rendered) + (2 if current else 11)
            if used + additional > max_characters:
                continue
            selected.setdefault(table_name, {}).setdefault(
                column_name,
                [],
            ).append(candidates[value_index])
            used += additional
    return selected


def serialize_schema(
    schema: DatabaseSchema,
    style: str,
    example_values: dict[str, dict[str, list[Any]]] | None = None,
    max_example_characters: int = DEFAULT_EXAMPLE_CHARACTER_BUDGET,
) -> str:
    """使用指定格式序列化 Schema。"""

    normalized_style = style.strip().lower()
    if normalized_style not in SUPPORTED_STYLES:
        supported = ", ".join(sorted(SUPPORTED_STYLES))
        raise ValueError(
            f"不支持 Schema 格式 {style!r}；可选值：{supported}"
        )

    if normalized_style == "ddl":
        return serialize_as_ddl(schema)
    if normalized_style == "structured_with_values":
        if example_values is None:
            raise ValueError(
                "structured_with_values 格式必须提供 example_values"
            )
        return serialize_as_structured(
            schema,
            example_values,
            max_example_characters=max_example_characters,
        )
    return serialize_as_structured(schema)
