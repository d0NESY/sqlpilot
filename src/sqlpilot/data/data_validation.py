"""Spider 训练前数据完整性检查。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema_parser import open_read_only_database, parse_sqlite_schema
from .spider_schema import (
    compare_spider_and_sqlite_schema,
    load_spider_schema_catalog,
)
from .training_data import load_spider_samples


REQUIRED_SAMPLE_FILES = {
    "train_spider": "train_spider.json",
    "train_others": "train_others.json",
    "dev": "dev.json",
}
KnownInvalidGold = dict[tuple[str, int], dict[str, Any]]


@dataclass(frozen=True)
class DataCheckResult:
    """训练前数据检查的三类结果。"""

    summary: dict[str, Any]
    invalid_samples: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]
    database_manifest: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_known_invalid_gold(
    path: str | Path | None,
) -> KnownInvalidGold:
    """读取版本化 Gold 排除清单，并验证键的唯一性。"""

    if path is None:
        return {}
    allowlist_path = Path(path)
    if not allowlist_path.is_file():
        raise FileNotFoundError(
            f"找不到 known invalid Gold 清单：{allowlist_path}"
        )
    with allowlist_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("version") != 1:
        raise ValueError("known invalid Gold 清单版本必须为 1")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("known invalid Gold 的 entries 必须是列表")

    result: KnownInvalidGold = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("known invalid Gold 条目必须是对象")
        source = entry.get("source")
        index = entry.get("index")
        if (
            source not in REQUIRED_SAMPLE_FILES
            or not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
        ):
            raise ValueError(f"非法 known invalid Gold 条目：{entry}")
        key = (source, index)
        if key in result:
            raise ValueError(f"known invalid Gold 重复键：{key}")
        result[key] = entry
    return result


def _invalid(
    source: str,
    index: int | None,
    db_id: Any,
    stage: str,
    error: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "index": index,
        "db_id": db_id,
        "stage": stage,
        "error": error,
    }


def _execute_gold_sql(
    connection: sqlite3.Connection,
    query: str,
    timeout_seconds: float,
) -> None:
    start = time.monotonic()

    def should_abort() -> int:
        return int(time.monotonic() - start > timeout_seconds)

    connection.set_progress_handler(should_abort, 10_000)
    try:
        connection.execute(query).fetchmany(1)
    finally:
        connection.set_progress_handler(None, 0)


def run_spider_data_check(
    data_root: str | Path,
    execute_gold: bool = True,
    query_timeout_seconds: float = 5.0,
    known_invalid_gold: KnownInvalidGold | None = None,
) -> DataCheckResult:
    """执行所有训练前数据检查，但不在发现首个错误时提前停止。"""

    root = Path(data_root)
    tables_path = root / "tables.json"
    database_root = root / "database"
    required_paths = {
        **{
            name: root / file_name
            for name, file_name in REQUIRED_SAMPLE_FILES.items()
        },
        "tables": tables_path,
        "database": database_root,
    }

    missing = [
        str(path) for path in required_paths.values() if not path.exists()
    ]
    if missing:
        summary = {
            "status": "failed",
            "missing_paths": missing,
            "invalid_count": len(missing),
        }
        return DataCheckResult(
            summary=summary,
            invalid_samples=tuple(
                _invalid("dataset", None, None, "missing_path", path)
                for path in missing
            ),
            warnings=(),
            database_manifest={
                "hash_algorithm": "sha256",
                "databases": [],
            },
        )

    invalid: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    exclusions = known_invalid_gold or {}
    catalog = load_spider_schema_catalog(tables_path)
    source_samples: dict[str, list[dict[str, Any]]] = {
        source: load_spider_samples(root / file_name)
        for source, file_name in REQUIRED_SAMPLE_FILES.items()
    }

    database_paths = {
        db_id: database_root / db_id / f"{db_id}.sqlite"
        for db_id in catalog
    }
    for db_id, path in database_paths.items():
        if not path.is_file():
            invalid.append(
                _invalid(
                    "database",
                    None,
                    db_id,
                    "missing_database",
                    str(path),
                )
            )

    source_db_ids: dict[str, set[str]] = {
        source: set() for source in source_samples
    }
    valid_samples: dict[str, list[tuple[int, dict[str, Any]]]] = {
        source: [] for source in source_samples
    }
    for source, samples in source_samples.items():
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                invalid.append(
                    _invalid(
                        source,
                        index,
                        None,
                        "sample_format",
                        "样本必须是对象",
                    )
                )
                continue
            errors: list[str] = []
            for field in ("question", "query", "db_id"):
                value = sample.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{field} 必须是非空字符串")
            db_id = sample.get("db_id")
            if isinstance(db_id, str) and db_id.strip():
                db_id = db_id.strip()
                source_db_ids[source].add(db_id)
                if db_id not in catalog:
                    errors.append("db_id 不存在于 tables.json")
                elif not database_paths[db_id].is_file():
                    errors.append("db_id 没有对应 SQLite 文件")
            for error in errors:
                invalid.append(
                    _invalid(
                        source,
                        index,
                        db_id,
                        "sample_fields",
                        error,
                    )
                )
            if not errors:
                valid_samples[source].append((index, sample))

    train_db_ids = (
        source_db_ids["train_spider"]
        | source_db_ids["train_others"]
    )
    dev_db_ids = source_db_ids["dev"]
    overlap = sorted(train_db_ids & dev_db_ids)
    if overlap:
        invalid.append(
            _invalid(
                "dataset",
                None,
                None,
                "data_leakage",
                f"train/dev 数据库重叠：{overlap}",
            )
        )

    schema_checked = 0
    schema_mismatch_count = 0
    schema_warning_count = 0
    schema_critical_count = 0
    for db_id, official in catalog.items():
        path = database_paths[db_id]
        if not path.is_file():
            continue
        try:
            sqlite_schema = parse_sqlite_schema(db_id, path)
            errors = compare_spider_and_sqlite_schema(
                official.database,
                sqlite_schema,
            )
            schema_checked += 1
            for error in errors:
                schema_mismatch_count += 1
                item = _invalid(
                    "tables.json",
                    None,
                    db_id,
                    "schema_mismatch",
                    error,
                )
                if error.startswith(("主键不一致", "外键不一致")):
                    schema_warning_count += 1
                    warnings.append(item)
                else:
                    schema_critical_count += 1
                    invalid.append(item)
        except Exception as error:  # 汇总检查必须继续处理其他数据库
            invalid.append(
                _invalid(
                    "database",
                    None,
                    db_id,
                    "sqlite_open",
                    f"{type(error).__name__}: {error}",
                )
            )

    gold_checked = 0
    gold_failed = 0
    gold_known_invalid = 0
    seen_exclusions: set[tuple[str, int]] = set()
    if execute_gold:
        connections: dict[str, sqlite3.Connection] = {}
        try:
            for source, samples in valid_samples.items():
                for index, sample in samples:
                    db_id = sample["db_id"].strip()
                    try:
                        if db_id not in connections:
                            connections[db_id] = open_read_only_database(
                                database_paths[db_id]
                            )
                        _execute_gold_sql(
                            connections[db_id],
                            sample["query"],
                            timeout_seconds=query_timeout_seconds,
                        )
                        gold_checked += 1
                    except Exception as error:
                        exclusion = exclusions.get((source, index))
                        query_hash = sha256_text(sample["query"])
                        if (
                            exclusion is not None
                            and exclusion.get("db_id") == db_id
                            and exclusion.get("query_sha256") == query_hash
                        ):
                            gold_known_invalid += 1
                            seen_exclusions.add((source, index))
                            warnings.append(
                                _invalid(
                                    source,
                                    index,
                                    db_id,
                                    "known_invalid_gold",
                                    exclusion.get("reason", str(error)),
                                )
                            )
                        else:
                            gold_failed += 1
                            invalid.append(
                                _invalid(
                                    source,
                                    index,
                                    db_id,
                                    "gold_execution",
                                    f"{type(error).__name__}: {error}",
                                )
                            )
        finally:
            for connection in connections.values():
                connection.close()

        missing_exclusions = sorted(set(exclusions) - seen_exclusions)
        for source, index in missing_exclusions:
            invalid.append(
                _invalid(
                    source,
                    index,
                    exclusions[(source, index)].get("db_id"),
                    "stale_gold_exclusion",
                    "排除清单条目未对应到本次失败；请核对数据版本和哈希",
                )
            )

    databases: list[dict[str, Any]] = []
    for db_id, path in sorted(database_paths.items()):
        if not path.is_file():
            continue
        if db_id in train_db_ids:
            split = "train"
        elif db_id in dev_db_ids:
            split = "dev"
        else:
            split = "unused"
        databases.append(
            {
                "db_id": db_id,
                "split": split,
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    source_files = {
        name: {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in required_paths.items()
        if name != "database" and path.is_file()
    }
    summary = {
        "status": "passed" if not invalid else "failed",
        "sample_counts": {
            source: len(samples)
            for source, samples in source_samples.items()
        },
        "total_samples": sum(map(len, source_samples.values())),
        "official_schema_count": len(catalog),
        "sqlite_schema_checked": schema_checked,
        "schema_mismatch_count": schema_mismatch_count,
        "schema_warning_count": schema_warning_count,
        "schema_critical_count": schema_critical_count,
        "train_database_count": len(train_db_ids),
        "dev_database_count": len(dev_db_ids),
        "train_dev_database_overlap": overlap,
        "gold_execution_enabled": execute_gold,
        "gold_execution_checked": gold_checked,
        "gold_execution_failed": gold_failed,
        "gold_known_invalid_excluded": gold_known_invalid,
        "invalid_count": len(invalid),
        "hash_algorithm": "sha256",
    }
    manifest = {
        "hash_algorithm": "sha256",
        "source_files": source_files,
        "databases": databases,
    }
    return DataCheckResult(
        summary=summary,
        invalid_samples=tuple(invalid),
        warnings=tuple(warnings),
        database_manifest=manifest,
    )


def write_json(path: str | Path, value: Any) -> None:
    """原子写入格式化 JSON。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(output)


def write_invalid_jsonl(
    path: str | Path,
    invalid_samples: tuple[dict[str, Any], ...],
) -> None:
    """原子写入无效样本；没有错误时生成空文件。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for sample in invalid_samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")
    temporary.replace(output)
