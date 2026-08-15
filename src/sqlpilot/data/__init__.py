"""SQLPilot 的数据读取与 Schema 处理模块。"""

from .schema_parser import (
    ColumnSchema,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
    parse_sqlite_schema,
)
from .data_validation import (
    DataCheckResult,
    load_known_invalid_gold,
    run_spider_data_check,
    sha256_file,
    write_invalid_jsonl,
    write_json,
)
from .dataset_preparation import (
    DATASET_VARIANTS,
    DatasetVariant,
    IndexedSpiderSample,
    load_indexed_samples,
    prepare_experiment_datasets,
    split_by_database,
)
from .schema_serializer import serialize_schema
from .schema_linker import (
    SchemaLink,
    extract_schema_link,
    serialize_schema_link,
)
from .spider_schema import (
    SpiderSchema,
    compare_spider_and_sqlite_schema,
    load_spider_schema_catalog,
    parse_spider_schema_entry,
)
from .training_data import (
    SpiderTrainingDataBuilder,
    TrainingRecord,
    load_spider_samples,
    normalize_gold_sql,
    write_jsonl,
)
from .value_sampler import (
    ExampleValues,
    load_or_create_value_cache,
    sample_database_values,
)

__all__ = [
    "ColumnSchema",
    "DataCheckResult",
    "DATASET_VARIANTS",
    "DatabaseSchema",
    "DatasetVariant",
    "ForeignKeySchema",
    "IndexedSpiderSample",
    "ExampleValues",
    "SchemaLink",
    "SpiderSchema",
    "TableSchema",
    "SpiderTrainingDataBuilder",
    "TrainingRecord",
    "compare_spider_and_sqlite_schema",
    "extract_schema_link",
    "load_known_invalid_gold",
    "load_indexed_samples",
    "load_or_create_value_cache",
    "load_spider_schema_catalog",
    "load_spider_samples",
    "normalize_gold_sql",
    "parse_spider_schema_entry",
    "parse_sqlite_schema",
    "prepare_experiment_datasets",
    "run_spider_data_check",
    "sample_database_values",
    "sha256_file",
    "split_by_database",
    "serialize_schema_link",
    "serialize_schema",
    "write_jsonl",
    "write_invalid_jsonl",
    "write_json",
]
