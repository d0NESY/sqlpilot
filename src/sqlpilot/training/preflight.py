"""训练数据身份、对话格式和 Token 长度预检。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlpilot.data import sha256_file


def _percentile(sorted_values: list[int], percentile: float) -> int:
    if not sorted_values:
        return 0
    index = math.ceil(percentile * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def validate_dataset_identity(
    path: str | Path,
    expected_records: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """先核对文件哈希和行数，防止训练了错误版本的数据。"""

    dataset_path = Path(path)
    digest = sha256_file(dataset_path)
    with dataset_path.open("r", encoding="utf-8") as file:
        record_count = sum(1 for line in file if line.strip())
    if digest != expected_sha256:
        raise ValueError(
            f"数据哈希不一致：{dataset_path}\n"
            f"expected={expected_sha256}\nactual={digest}"
        )
    if record_count != expected_records:
        raise ValueError(
            f"数据行数不一致：{dataset_path}，"
            f"expected={expected_records}, actual={record_count}"
        )
    return {
        "path": str(dataset_path),
        "record_count": record_count,
        "sha256": digest,
    }


def _validate_messages(
    messages: Any,
    expected_roles: tuple[str, ...],
    sample_id: str,
) -> list[dict[str, str]]:
    if not isinstance(messages, list) or len(messages) != len(expected_roles):
        raise ValueError(f"{sample_id} 的消息数量不正确")
    for message, expected_role in zip(messages, expected_roles):
        if (
            not isinstance(message, dict)
            or message.get("role") != expected_role
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError(
                f"{sample_id} 的 {expected_role} 消息格式不正确"
            )
    return messages


def _chat_token_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> list[int]:
    """兼容当前返回 BatchEncoding 和旧版直接返回 ID 列表的接口。"""

    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    if isinstance(encoded, Mapping):
        token_ids = encoded["input_ids"]
    else:
        token_ids = encoded
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return list(token_ids)


def analyze_jsonl(
    path: str | Path,
    tokenizer: Any,
    max_length: int,
    max_offenders: int = 100,
) -> dict[str, Any]:
    """使用模型 chat template 统计完整序列和 completion 长度。"""

    lengths: list[int] = []
    completion_lengths: list[int] = []
    over_max: list[dict[str, Any]] = []
    prompt_reaches_max: list[dict[str, Any]] = []
    over_max_count = 0
    prompt_reaches_max_count = 0
    eos_missing = 0

    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id", f"line_{line_number}")
            prompt = _validate_messages(
                record.get("prompt"),
                ("system", "user"),
                sample_id,
            )
            completion = _validate_messages(
                record.get("completion"),
                ("assistant",),
                sample_id,
            )
            full_ids = _chat_token_ids(
                tokenizer,
                prompt + completion,
                add_generation_prompt=False,
            )
            prompt_ids = _chat_token_ids(
                tokenizer,
                prompt,
                add_generation_prompt=True,
            )
            full_length = len(full_ids)
            prompt_length = len(prompt_ids)
            completion_length = max(0, full_length - prompt_length)
            lengths.append(full_length)
            completion_lengths.append(completion_length)

            item = {
                "sample_id": sample_id,
                "db_id": record.get("db_id"),
                "full_length": full_length,
                "prompt_length": prompt_length,
                "completion_length": completion_length,
            }
            if full_length > max_length and len(over_max) < max_offenders:
                over_max.append(item)
            if full_length > max_length:
                over_max_count += 1
            if (
                prompt_length >= max_length
                and len(prompt_reaches_max) < max_offenders
            ):
                prompt_reaches_max.append(item)
            if prompt_length >= max_length:
                prompt_reaches_max_count += 1
            if (
                tokenizer.eos_token_id is not None
                and tokenizer.eos_token_id not in full_ids[-4:]
            ):
                eos_missing += 1

    sorted_lengths = sorted(lengths)
    sorted_completion = sorted(completion_lengths)
    return {
        "path": str(Path(path)),
        "record_count": len(lengths),
        "max_length_setting": max_length,
        "full_length": {
            "min": sorted_lengths[0] if sorted_lengths else 0,
            "p50": _percentile(sorted_lengths, 0.50),
            "p95": _percentile(sorted_lengths, 0.95),
            "p99": _percentile(sorted_lengths, 0.99),
            "max": sorted_lengths[-1] if sorted_lengths else 0,
        },
        "completion_length": {
            "min": sorted_completion[0] if sorted_completion else 0,
            "p50": _percentile(sorted_completion, 0.50),
            "p95": _percentile(sorted_completion, 0.95),
            "max": sorted_completion[-1] if sorted_completion else 0,
        },
        "over_max_count": over_max_count,
        "prompt_reaches_max_count": prompt_reaches_max_count,
        "eos_missing_count": eos_missing,
        "over_max_examples": over_max,
        "prompt_reaches_max_examples": prompt_reaches_max,
    }


def enforce_length_safety(
    report: dict[str, Any],
    allow_truncation: bool,
) -> None:
    """默认拒绝任何截断；即使放宽，也绝不允许 Prompt 吃掉答案。"""

    if report["prompt_reaches_max_count"]:
        raise ValueError(
            f"{report['path']} 有 Prompt 达到 max_length，"
            "completion 会完全丢失"
        )
    if report["over_max_count"] and not allow_truncation:
        raise ValueError(
            f"{report['path']} 有 {report['over_max_count']} 条超过 "
            "max_length；请增大长度或调整 Schema，禁止静默截断"
        )
    if report["eos_missing_count"]:
        raise ValueError(
            f"{report['path']} 有 {report['eos_missing_count']} 条"
            "完整对话未以 EOS 结束，请检查 chat template"
        )
