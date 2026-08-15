"""从当前 SQLite 数据库确定性采样少量列值，并写入缓存。"""

from __future__ import annotations

import json
import re
from contextlib import closing
from pathlib import Path
from typing import Any

from .schema_parser import (
    DatabaseSchema,
    open_read_only_database,
    quote_identifier,
)


VALUE_CACHE_VERSION = 1
SENSITIVE_COLUMN_PATTERN = re.compile(
    r"(?:password|passwd|secret|token|api[_ -]?key|"
    r"e[_ -]?mail|phone|mobile|ssn|credit[_ -]?card)",
    re.IGNORECASE,
)

ExampleValues = dict[str, dict[str, list[Any]]]


def _normalized_value(value: Any, max_text_length: int) -> Any | None:
    if value is None or isinstance(value, bytes):
        return None
    if isinstance(value, str):
        cleaned = value.replace("\x00", "").strip()
        if not cleaned:
            return None
        return cleaned[:max_text_length]
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    return text[:max_text_length] if text else None


def sample_database_values(
    schema: DatabaseSchema,
    db_path: str | Path,
    max_values_per_column: int = 3,
    max_text_length: int = 30,
) -> ExampleValues:
    """按文本排序取每列前几个非空不同值，结果可复现。"""

    if max_values_per_column <= 0:
        raise ValueError("max_values_per_column 必须大于 0")
    if max_text_length <= 0:
        raise ValueError("max_text_length 必须大于 0")

    values: ExampleValues = {}
    with closing(open_read_only_database(Path(db_path))) as connection:
        for table in schema.tables:
            table_values: dict[str, list[Any]] = {}
            quoted_table = quote_identifier(table.name)
            for column in table.columns:
                if "BLOB" in column.data_type.upper():
                    continue
                if SENSITIVE_COLUMN_PATTERN.search(column.name):
                    continue

                quoted_column = quote_identifier(column.name)
                query = (
                    f"SELECT DISTINCT {quoted_column} "
                    f"FROM {quoted_table} "
                    f"WHERE {quoted_column} IS NOT NULL "
                    f"ORDER BY CAST({quoted_column} AS TEXT) COLLATE BINARY "
                    f"LIMIT ?;"
                )
                rows = connection.execute(
                    query,
                    (max_values_per_column * 4,),
                ).fetchall()
                sampled: list[Any] = []
                for row in rows:
                    value = _normalized_value(row[0], max_text_length)
                    if value is None or value in sampled:
                        continue
                    sampled.append(value)
                    if len(sampled) >= max_values_per_column:
                        break
                if sampled:
                    table_values[column.name] = sampled
            values[table.name] = table_values
    return values


def load_or_create_value_cache(
    schema: DatabaseSchema,
    db_path: str | Path,
    cache_root: str | Path,
    max_values_per_column: int = 3,
    max_text_length: int = 30,
    force: bool = False,
) -> ExampleValues:
    """读取匹配当前参数的缓存；没有缓存时采样并原子写入。"""

    cache_path = Path(cache_root) / f"{schema.db_id}.json"
    settings = {
        "max_values_per_column": max_values_per_column,
        "max_text_length": max_text_length,
    }
    if cache_path.is_file() and not force:
        with cache_path.open("r", encoding="utf-8") as file:
            cached = json.load(file)
        if (
            cached.get("version") == VALUE_CACHE_VERSION
            and cached.get("db_id") == schema.db_id
            and cached.get("settings") == settings
            and isinstance(cached.get("values"), dict)
        ):
            return cached["values"]

    values = sample_database_values(
        schema=schema,
        db_path=db_path,
        max_values_per_column=max_values_per_column,
        max_text_length=max_text_length,
    )
    payload = {
        "version": VALUE_CACHE_VERSION,
        "db_id": schema.db_id,
        "settings": settings,
        "values": values,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(cache_path)
    return values
