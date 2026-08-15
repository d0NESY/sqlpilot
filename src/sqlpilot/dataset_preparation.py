"""生成 S1-S4 数据集和训练前小规模闸门数据。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .data_validation import (
    KnownInvalidGold,
    sha256_file,
    sha256_text,
    write_json,
)
from .spider_schema import SpiderSchema
from .schema_serializer import DEFAULT_EXAMPLE_CHARACTER_BUDGET
from .training_data import (
    SpiderTrainingDataBuilder,
    load_spider_samples,
    write_jsonl,
)
from .value_sampler import VALUE_CACHE_VERSION


@dataclass(frozen=True)
class IndexedSpiderSample:
    """保留来源和原始下标的 Spider 样本。"""

    source: str
    index: int
    sample: dict[str, Any]

    @property
    def sample_id(self) -> str:
        return f"{self.source}_{self.index + 1:06d}"

    @property
    def db_id(self) -> str:
        return self.sample["db_id"].strip()


@dataclass(frozen=True)
class DatasetVariant:
    """一个实验的数据表示。"""

    name: str
    schema_style: str
    output_mode: str


DATASET_VARIANTS = (
    DatasetVariant("s1_ddl", "ddl", "direct"),
    DatasetVariant("s2_structured", "structured", "direct"),
    DatasetVariant(
        "s3_values",
        "structured_with_values",
        "direct",
    ),
    DatasetVariant(
        "s4_schema_link",
        "structured_with_values",
        "schema_link",
    ),
)


def load_indexed_samples(
    source_paths: Mapping[str, str | Path],
    exclusions: KnownInvalidGold | None = None,
) -> tuple[list[IndexedSpiderSample], list[dict[str, Any]]]:
    """读取样本并按来源、下标、db_id、查询哈希精确排除异常 Gold。"""

    excluded_entries = exclusions or {}
    samples: list[IndexedSpiderSample] = []
    excluded: list[dict[str, Any]] = []
    seen_exclusions: set[tuple[str, int]] = set()

    for source, path in source_paths.items():
        for index, sample in enumerate(load_spider_samples(path)):
            exclusion = excluded_entries.get((source, index))
            if exclusion is not None:
                query = sample.get("query")
                db_id = sample.get("db_id")
                if (
                    not isinstance(query, str)
                    or exclusion.get("query_sha256") != sha256_text(query)
                    or exclusion.get("db_id") != db_id
                ):
                    raise ValueError(
                        f"排除清单与当前数据不匹配：{source}[{index}]"
                    )
                excluded.append(exclusion)
                seen_exclusions.add((source, index))
                continue
            samples.append(
                IndexedSpiderSample(
                    source=source,
                    index=index,
                    sample=sample,
                )
            )

    unseen = sorted(set(excluded_entries) - seen_exclusions)
    if unseen:
        raise ValueError(f"以下排除项没有对应样本：{unseen}")
    return samples, excluded


def split_by_database(
    samples: Iterable[IndexedSpiderSample],
    validation_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[
    list[IndexedSpiderSample],
    list[IndexedSpiderSample],
    tuple[str, ...],
]:
    """按数据库切分，保证内部 train/validation 不共享数据库。"""

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio 必须在 0 和 1 之间")
    items = list(samples)
    database_ids = sorted({item.db_id for item in items})
    if len(database_ids) < 2:
        raise ValueError("至少需要两个数据库才能按数据库切分")

    shuffled = list(database_ids)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(
        1,
        min(len(shuffled) - 1, round(len(shuffled) * validation_ratio)),
    )
    validation_databases = frozenset(shuffled[:validation_count])
    train = [
        item for item in items if item.db_id not in validation_databases
    ]
    validation = [
        item for item in items if item.db_id in validation_databases
    ]
    return train, validation, tuple(sorted(validation_databases))


def _records(
    builder: SpiderTrainingDataBuilder,
    samples: Iterable[IndexedSpiderSample],
):
    for item in samples:
        yield builder.build_record(
            sample=item.sample,
            sample_id=item.sample_id,
        )


def _file_metadata(
    path: Path,
    relative_to: Path,
    count: int,
    role: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "relative_path": path.relative_to(relative_to).as_posix(),
        "record_count": count,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def prepare_experiment_datasets(
    data_root: str | Path,
    output_root: str | Path,
    value_cache_root: str | Path,
    schema_catalog: Mapping[str, SpiderSchema],
    train_samples: list[IndexedSpiderSample],
    validation_samples: list[IndexedSpiderSample],
    dev_samples: list[IndexedSpiderSample],
    validation_database_ids: tuple[str, ...],
    excluded_gold: list[dict[str, Any]],
    split_seed: int = 42,
    validation_ratio: float = 0.10,
) -> dict[str, Any]:
    """生成 S1-S4、overfit16 和 smoke 数据并返回清单。"""

    raw_root = Path(data_root)
    processed_root = Path(output_root)
    database_root = raw_root / "database"
    cache_root = Path(value_cache_root)
    variant_manifests: dict[str, Any] = {}

    for variant in DATASET_VARIANTS:
        variant_root = processed_root / variant.name
        builder = SpiderTrainingDataBuilder(
            database_root=database_root,
            schema_style=variant.schema_style,
            official_schema_catalog=schema_catalog,
            value_cache_root=cache_root,
            output_mode=variant.output_mode,
        )
        files: list[dict[str, Any]] = []
        for file_name, samples, role in (
            ("train.jsonl", train_samples, "train"),
            (
                "validation.jsonl",
                validation_samples,
                "internal_validation",
            ),
            (
                "official_dev_evaluation_only.jsonl",
                dev_samples,
                "official_evaluation_only",
            ),
        ):
            path = variant_root / file_name
            count = write_jsonl(_records(builder, samples), path)
            files.append(
                _file_metadata(
                    path,
                    processed_root.parent,
                    count,
                    role,
                )
            )
        variant_manifests[variant.name] = {
            "schema_style": variant.schema_style,
            "output_mode": variant.output_mode,
            "files": files,
        }

    gate_root = processed_root / "gates"
    gate_builder = SpiderTrainingDataBuilder(
        database_root=database_root,
        schema_style="structured",
        official_schema_catalog=schema_catalog,
        output_mode="direct",
    )
    gate_specs = (
        ("overfit16.jsonl", train_samples[:16], "overfit_gate"),
        ("smoke_train_100.jsonl", train_samples[:100], "smoke_train"),
        (
            "smoke_validation_20.jsonl",
            validation_samples[:20],
            "smoke_validation",
        ),
    )
    gate_files: list[dict[str, Any]] = []
    for file_name, samples, role in gate_specs:
        path = gate_root / file_name
        count = write_jsonl(_records(gate_builder, samples), path)
        gate_files.append(
            _file_metadata(
                path,
                processed_root.parent,
                count,
                role,
            )
        )

    train_database_ids = sorted({item.db_id for item in train_samples})
    dev_database_ids = sorted({item.db_id for item in dev_samples})
    manifest = {
        "format_version": 1,
        "split_strategy": "database_level",
        "split_seed": split_seed,
        "validation_ratio": validation_ratio,
        "train_record_count": len(train_samples),
        "validation_record_count": len(validation_samples),
        "official_dev_record_count": len(dev_samples),
        "train_database_count": len(train_database_ids),
        "validation_database_count": len(validation_database_ids),
        "official_dev_database_count": len(dev_database_ids),
        "validation_database_ids": list(validation_database_ids),
        "official_dev_is_evaluation_only": True,
        "excluded_gold": excluded_gold,
        "example_value_policy": {
            "cache_version": VALUE_CACHE_VERSION,
            "max_values_per_column": 3,
            "max_text_length": 30,
            "sensitive_columns_skipped": True,
            "prompt_character_budget_per_database": (
                DEFAULT_EXAMPLE_CHARACTER_BUDGET
            ),
            "budget_allocation": "column_round_robin",
        },
        "variants": variant_manifests,
        "gates": gate_files,
    }
    write_json(processed_root / "dataset_manifest.json", manifest)
    return manifest
