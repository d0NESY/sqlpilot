"""把 Spider 样本转换成可用于监督微调的 Prompt-Completion 数据。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, TypedDict

from .schema_parser import parse_sqlite_schema
from .schema_linker import extract_schema_link, serialize_schema_link
from .schema_serializer import serialize_schema
from .spider_schema import SpiderSchema
from .value_sampler import load_or_create_value_cache


SYSTEM_MESSAGE = (
    "You are an expert SQLite developer. "
    "Generate exactly one executable read-only SQLite query using only "
    "the provided schema. Do not invent identifiers. Return no explanation "
    "or Markdown. Return one statement only and end it with a semicolon."
)
SCHEMA_LINK_SYSTEM_MESSAGE = (
    "You are an expert SQLite developer. Use only the provided schema. "
    "First return the relevant tables and columns inside a <SCHEMA_LINK> "
    "block. Then return exactly one executable read-only SQLite query "
    "inside a <SQL> block. Do not invent identifiers or add explanations."
)
SUPPORTED_OUTPUT_MODES = {"direct", "schema_link"}


class ChatMessage(TypedDict):
    """一条对话消息。"""

    role: str
    content: str


class TrainingRecord(TypedDict):
    """写入 JSONL 的一条训练记录。"""

    prompt: list[ChatMessage]
    completion: list[ChatMessage]
    db_id: str
    sample_id: str


def load_spider_samples(json_path: str | Path) -> list[dict[str, Any]]:
    """读取 Spider JSON 文件，并检查最外层是不是样本列表。"""

    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 Spider 样本文件：{path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Spider 文件最外层必须是列表：{path}")
    return data


def normalize_gold_sql(sql: str) -> str:
    """保留 Gold SQL 内容，并统一补上一个结尾分号。"""

    normalized = sql.strip()
    if not normalized:
        raise ValueError("Gold SQL 不能为空")
    return normalized.rstrip(";").rstrip() + ";"


def _required_text(
    sample: dict[str, Any],
    field_name: str,
    sample_id: str,
) -> str:
    value = sample.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"样本 {sample_id} 的 {field_name!r} 必须是非空字符串"
        )
    return value.strip()


class SpiderTrainingDataBuilder:
    """使用数据库 Schema 构造 Spider Prompt-Completion 记录。"""

    def __init__(
        self,
        database_root: str | Path,
        schema_style: str = "structured",
        official_schema_catalog: Mapping[str, SpiderSchema] | None = None,
        value_cache_root: str | Path | None = None,
        output_mode: str = "direct",
    ) -> None:
        self.database_root = Path(database_root)
        self.schema_style = schema_style
        self.official_schema_catalog = official_schema_catalog
        self.value_cache_root = (
            Path(value_cache_root) if value_cache_root is not None else None
        )
        self.output_mode = output_mode.strip().lower()
        if self.output_mode not in SUPPORTED_OUTPUT_MODES:
            raise ValueError(
                f"output_mode 必须是 {sorted(SUPPORTED_OUTPUT_MODES)}"
            )
        if (
            self.schema_style == "structured_with_values"
            and self.value_cache_root is None
        ):
            raise ValueError(
                "structured_with_values 必须提供 value_cache_root"
            )
        if (
            self.output_mode == "schema_link"
            and self.official_schema_catalog is None
        ):
            raise ValueError(
                "schema_link 输出必须提供 official_schema_catalog"
            )
        self._schema_cache: dict[str, str] = {}

    @property
    def cached_database_count(self) -> int:
        """已经解析并缓存了多少个数据库。"""

        return len(self._schema_cache)

    def _database_path(self, db_id: str) -> Path:
        return self.database_root / db_id / f"{db_id}.sqlite"

    def _serialized_schema(self, db_id: str) -> str:
        if db_id not in self._schema_cache:
            if self.official_schema_catalog is not None:
                try:
                    spider_schema = self.official_schema_catalog[db_id]
                except KeyError as error:
                    raise ValueError(
                        f"tables.json 中找不到 db_id：{db_id}"
                    ) from error
                schema = spider_schema.database
            else:
                schema = parse_sqlite_schema(
                    db_id=db_id,
                    db_path=self._database_path(db_id),
                )

            example_values = None
            if self.schema_style == "structured_with_values":
                example_values = load_or_create_value_cache(
                    schema=schema,
                    db_path=self._database_path(db_id),
                    cache_root=self.value_cache_root,
                )
            self._schema_cache[db_id] = serialize_schema(
                schema,
                style=self.schema_style,
                example_values=example_values,
            )
        return self._schema_cache[db_id]

    def _completion(
        self,
        sample: dict[str, Any],
        db_id: str,
        gold_sql: str,
        sample_id: str,
    ) -> str:
        if self.output_mode == "direct":
            return gold_sql

        sql_ast = sample.get("sql")
        if not isinstance(sql_ast, dict):
            raise ValueError(
                f"样本 {sample_id} 缺少 Spider 结构化 sql 字段"
            )
        assert self.official_schema_catalog is not None
        schema = self.official_schema_catalog[db_id]
        link = serialize_schema_link(extract_schema_link(sql_ast, schema))
        return f"{link}\n<SQL>\n{gold_sql}\n</SQL>"

    def build_record(
        self,
        sample: dict[str, Any],
        sample_id: str,
    ) -> TrainingRecord:
        """构造一条训练记录。"""

        db_id = _required_text(sample, "db_id", sample_id)
        question = _required_text(sample, "question", sample_id)
        gold_sql = normalize_gold_sql(
            _required_text(sample, "query", sample_id)
        )
        schema_text = self._serialized_schema(db_id)
        completion = self._completion(
            sample=sample,
            db_id=db_id,
            gold_sql=gold_sql,
            sample_id=sample_id,
        )

        user_message = (
            f"Database schema:\n{schema_text}\n\n"
            f"Question:\n{question}"
        )

        return {
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        SYSTEM_MESSAGE
                        if self.output_mode == "direct"
                        else SCHEMA_LINK_SYSTEM_MESSAGE
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            "completion": [
                {"role": "assistant", "content": completion},
            ],
            "db_id": db_id,
            "sample_id": sample_id,
        }

    def build_records(
        self,
        samples: Iterable[dict[str, Any]],
        split: str,
        limit: int | None = None,
    ) -> Iterable[TrainingRecord]:
        """按原始顺序逐条生成记录，sample_id 从 000001 开始。"""

        if not split.strip():
            raise ValueError("split 不能为空")
        if limit is not None and limit < 0:
            raise ValueError("limit 不能小于 0")

        for index, sample in enumerate(samples, start=1):
            if limit is not None and index > limit:
                break
            sample_id = f"{split}_{index:06d}"
            yield self.build_record(sample, sample_id=sample_id)


def write_jsonl(
    records: Iterable[TrainingRecord],
    output_path: str | Path,
) -> int:
    """以 UTF-8 JSONL 格式写出记录，返回记录数量。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    temporary_path.replace(path)
    return count
