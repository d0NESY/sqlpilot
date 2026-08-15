"""模型输出 SQL 提取、预测记录校验与官方文本格式转换。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


INVALID_SQL = "SELECT * FROM __sqlpilot_invalid_prediction__;"
_SQL_TAG = re.compile(r"<SQL>\s*(.*?)\s*</SQL>", re.IGNORECASE | re.DOTALL)
_SQL_FENCE = re.compile(
    r"```(?:sql|sqlite)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL
)
_SQL_START = re.compile(r"\b(?:SELECT|WITH)\b", re.IGNORECASE)
_FORBIDDEN_SQL = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExtractedSQL:
    sql: str
    method: str
    format_valid: bool


def _first_statement(text: str) -> str:
    """截取第一个分号结束的语句，同时保留字符串中的分号。"""

    quote: str | None = None
    bracket = False
    index = 0
    while index < len(text):
        character = text[index]
        if bracket:
            if character == "]":
                bracket = False
        elif quote is not None:
            if character == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    index += 1
                else:
                    quote = None
        elif character in {"'", '"', "`"}:
            quote = character
        elif character == "[":
            bracket = True
        elif character == ";":
            return text[: index + 1]
        index += 1
    return text


def _normalize_statement(candidate: str) -> str:
    candidate = candidate.strip()
    start = _SQL_START.search(candidate)
    if start is None:
        return ""
    statement = _first_statement(candidate[start.start() :]).strip()
    if not statement:
        return ""
    statement = " ".join(statement.splitlines())
    masked = re.sub(r"'(?:''|[^'])*'", "''", statement)
    masked = re.sub(r'"(?:""|[^"])*"', '""', masked)
    if _FORBIDDEN_SQL.search(masked):
        return ""
    return statement if statement.endswith(";") else statement + ";"


def extract_sql(raw_output: str, output_mode: str) -> ExtractedSQL:
    """以固定优先级提取一条只读 SQL，不修改查询语义。"""

    if not isinstance(raw_output, str) or not raw_output.strip():
        return ExtractedSQL(INVALID_SQL, "empty", False)

    tagged = _SQL_TAG.search(raw_output)
    if tagged is not None:
        sql = _normalize_statement(tagged.group(1))
        if sql:
            return ExtractedSQL(sql, "sql_tag", True)

    fenced = _SQL_FENCE.search(raw_output)
    if fenced is not None:
        sql = _normalize_statement(fenced.group(1))
        if sql:
            return ExtractedSQL(
                sql,
                "markdown_fence",
                output_mode == "direct",
            )

    sql = _normalize_statement(raw_output)
    if sql:
        return ExtractedSQL(
            sql,
            "plain_sql",
            output_mode == "direct",
        )
    return ExtractedSQL(INVALID_SQL, "unparseable", False)


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"预测第 {line_number} 行必须是对象")
            sample_id = record.get("sample_id")
            predicted_sql = record.get("predicted_sql")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"预测第 {line_number} 行缺少 sample_id")
            if sample_id in seen_ids:
                raise ValueError(f"预测 sample_id 重复：{sample_id}")
            if not isinstance(predicted_sql, str) or "\n" in predicted_sql:
                raise ValueError(
                    f"{sample_id} 的 predicted_sql 必须是单行字符串"
                )
            seen_ids.add(sample_id)
            records.append(record)
    return records


def materialize_official_predictions(
    records: Iterable[dict[str, Any]],
    expected_sample_ids: list[str],
    output_path: str | Path,
) -> Path:
    """严格按官方 dev 顺序写出每行一条 SQL 的预测文件。"""

    items = list(records)
    actual_ids = [record["sample_id"] for record in items]
    if actual_ids != expected_sample_ids:
        raise ValueError(
            "预测顺序或样本集合与官方 dev 不一致；拒绝生成评估文件"
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for record in items:
            sql = record["predicted_sql"].strip() or INVALID_SQL
            file.write(" ".join(sql.splitlines()) + "\n")
    temporary.replace(output)
    return output
