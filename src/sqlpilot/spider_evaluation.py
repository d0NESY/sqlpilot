"""准备 Spider 官方 dev Gold 文件并核对四种 Prompt 变体。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .data_validation import sha256_file, sha256_text, write_json
from .prediction import extract_sql
from .training_data import load_spider_samples, normalize_gold_sql


def _portable_path(path: Path) -> str:
    """优先记录相对项目路径，保证清单上传 Linux 后仍可读。"""

    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} 必须是对象")
            records.append(record)
    return records


def _gold_from_record(record: dict[str, Any], output_mode: str) -> str:
    completion = record.get("completion")
    if (
        not isinstance(completion, list)
        or len(completion) != 1
        or not isinstance(completion[0], dict)
        or not isinstance(completion[0].get("content"), str)
    ):
        raise ValueError(f"{record.get('sample_id')} completion 格式错误")
    extracted = extract_sql(completion[0]["content"], output_mode)
    if extracted.sql.startswith("SELECT * FROM __sqlpilot_invalid"):
        raise ValueError(f"{record.get('sample_id')} 无法提取 Gold SQL")
    return extracted.sql


def prepare_spider_dev_evaluation(
    raw_dev_path: str | Path,
    variant_paths: Mapping[str, str | Path],
    output_root: str | Path,
) -> dict[str, Any]:
    """生成官方 Gold/样本清单，并证明所有变体对应同一批 dev。"""

    raw_path = Path(raw_dev_path).resolve()
    raw_samples = load_spider_samples(raw_path)
    if len(raw_samples) != 1034:
        raise ValueError(f"Spider 官方 dev 应为 1034 条，实际 {len(raw_samples)}")

    variants: dict[str, dict[str, Any]] = {}
    expected_ids = [f"dev_{index:06d}" for index in range(1, 1035)]
    for name, value in variant_paths.items():
        path = Path(value).resolve()
        records = _load_jsonl(path)
        if len(records) != len(raw_samples):
            raise ValueError(f"{name} 记录数不是 1034：{len(records)}")
        output_mode = "schema_link" if "schema_link" in name else "direct"
        for index, (raw, record) in enumerate(
            zip(raw_samples, records), start=1
        ):
            sample_id = expected_ids[index - 1]
            if record.get("sample_id") != sample_id:
                raise ValueError(f"{name} 第 {index} 条 sample_id 不一致")
            if record.get("db_id") != raw.get("db_id"):
                raise ValueError(f"{name} 第 {index} 条 db_id 不一致")
            expected_gold = normalize_gold_sql(raw["query"])
            if _gold_from_record(record, output_mode) != expected_gold:
                raise ValueError(f"{name} 第 {index} 条 Gold SQL 不一致")
        variants[name] = {
            "path": _portable_path(path),
            "record_count": len(records),
            "sha256": sha256_file(path),
        }

    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    gold_path = output / "gold.sql"
    gold_tmp = gold_path.with_suffix(".sql.tmp")
    sample_path = output / "samples.jsonl"
    sample_tmp = sample_path.with_suffix(".jsonl.tmp")
    database_ids: set[str] = set()
    with (
        gold_tmp.open("w", encoding="utf-8", newline="\n") as gold_file,
        sample_tmp.open("w", encoding="utf-8", newline="\n") as sample_file,
    ):
        for index, sample in enumerate(raw_samples, start=1):
            query = sample["query"].strip().replace("\t", " ").replace("\n", " ")
            db_id = sample["db_id"].strip()
            sample_id = expected_ids[index - 1]
            database_ids.add(db_id)
            gold_file.write(f"{query}\t{db_id}\n")
            sample_file.write(
                json.dumps(
                    {
                        "index": index - 1,
                        "sample_id": sample_id,
                        "db_id": db_id,
                        "question": sample["question"],
                        "gold_sql_sha256": sha256_text(
                            normalize_gold_sql(sample["query"])
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    gold_tmp.replace(gold_path)
    sample_tmp.replace(sample_path)

    manifest = {
        "format_version": 1,
        "dataset": "Spider 1.0 official dev",
        "record_count": len(raw_samples),
        "database_count": len(database_ids),
        "sample_id_first": expected_ids[0],
        "sample_id_last": expected_ids[-1],
        "raw_dev": {
            "path": _portable_path(raw_path),
            "sha256": sha256_file(raw_path),
        },
        "variants": variants,
        "gold_file": {
            "path": _portable_path(gold_path),
            "sha256": sha256_file(gold_path),
        },
        "samples_file": {
            "path": _portable_path(sample_path),
            "sha256": sha256_file(sample_path),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest
