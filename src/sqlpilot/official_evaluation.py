"""调用固定版本的 Spider 原始与 Test Suite 官方评估器。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .data_validation import sha256_file, write_json
from .prediction import load_prediction_records, materialize_official_predictions


_METRIC_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _load_sample_ids(path: Path) -> list[str]:
    sample_ids: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                sample_ids.append(json.loads(line)["sample_id"])
    return sample_ids


def _prepare_runtime(source_root: Path, runtime_root: Path) -> dict[str, str]:
    """只复制根目录 Python 文件，并仅提高打印精度，不改算法。"""

    if not (source_root / "evaluation.py").is_file():
        raise FileNotFoundError(f"官方评估器缺少 evaluation.py：{source_root}")
    runtime_root.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for source in source_root.glob("*.py"):
        target = runtime_root / source.name
        content = source.read_text(encoding="utf-8")
        if source.name == "evaluation.py":
            content = content.replace("<20.3f", "<20.12f")
        target.write_text(content, encoding="utf-8", newline="\n")
        copied[source.name] = sha256_file(source)
    return copied


def _run(
    command: list[str],
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    combined = completed.stdout
    if completed.stderr:
        combined += "\n[stderr]\n" + completed.stderr
    log_path.write_text(combined, encoding="utf-8", newline="\n")
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout_log": str(log_path),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"官方评估失败（returncode={completed.returncode}），"
            f"详见 {log_path}"
        )
    return result


def _overall_metric(output: str, label: str) -> float:
    matches: list[float] = []
    for line in output.splitlines():
        stripped = line.strip().lower()
        if not stripped.startswith(label):
            continue
        numbers = [float(value) for value in _METRIC_NUMBER.findall(line)]
        if numbers:
            matches.append(numbers[-1])
    if not matches:
        raise ValueError(f"官方输出中找不到指标行：{label}")
    return matches[0]


def evaluate_official_predictions(
    predictions_jsonl: str | Path,
    sample_manifest_jsonl: str | Path,
    gold_path: str | Path,
    original_database_root: str | Path,
    tables_path: str | Path,
    spider_evaluator_root: str | Path,
    test_suite_evaluator_root: str | Path,
    test_suite_database_root: str | Path,
    output_root: str | Path,
    timeout_seconds: int = 21_600,
) -> dict[str, Any]:
    """产生官方 pred.sql，运行 EM/原始 EX/TSA，并写精确百分比。"""

    paths = {
        "predictions": Path(predictions_jsonl).resolve(),
        "samples": Path(sample_manifest_jsonl).resolve(),
        "gold": Path(gold_path).resolve(),
        "original_db": Path(original_database_root).resolve(),
        "tables": Path(tables_path).resolve(),
        "spider_eval": Path(spider_evaluator_root).resolve(),
        "test_suite_eval": Path(test_suite_evaluator_root).resolve(),
        "test_suite_db": Path(test_suite_database_root).resolve(),
    }
    for name, path in paths.items():
        if name.endswith("db") or name.endswith("eval"):
            exists = path.is_dir()
        else:
            exists = path.is_file()
        if not exists:
            raise FileNotFoundError(f"{name} 路径不存在：{path}")

    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_ids = _load_sample_ids(paths["samples"])
    if len(sample_ids) != 1034:
        raise ValueError(f"官方 dev 样本清单应为 1034 条，实际 {len(sample_ids)}")
    prediction_records = load_prediction_records(paths["predictions"])
    official_pred = materialize_official_predictions(
        prediction_records,
        sample_ids,
        output / "pred.sql",
    )

    runtime_root = output / "official_runtime"
    original_runtime = runtime_root / "spider"
    tsa_runtime = runtime_root / "test_suite"
    source_hashes = {
        "spider": _prepare_runtime(paths["spider_eval"], original_runtime),
        "test_suite": _prepare_runtime(paths["test_suite_eval"], tsa_runtime),
    }

    original_command = [
        sys.executable,
        "evaluation.py",
        "--gold",
        str(paths["gold"]),
        "--pred",
        str(official_pred),
        "--etype",
        "all",
        "--db",
        str(paths["original_db"]),
        "--table",
        str(paths["tables"]),
    ]
    original_log = output / "official_spider_em_ex.log"
    original_result = _run(
        original_command,
        original_runtime,
        original_log,
        timeout_seconds,
    )
    original_text = original_log.read_text(encoding="utf-8")

    tsa_command = [
        sys.executable,
        "evaluation.py",
        "--gold",
        str(paths["gold"]),
        "--pred",
        str(official_pred),
        "--etype",
        "exec",
        "--db",
        str(paths["test_suite_db"]),
    ]
    tsa_log = output / "official_test_suite_tsa.log"
    tsa_result = _run(
        tsa_command,
        tsa_runtime,
        tsa_log,
        timeout_seconds,
    )
    tsa_text = tsa_log.read_text(encoding="utf-8")

    em = _overall_metric(original_text, "exact match")
    ex = _overall_metric(original_text, "execution")
    tsa = _overall_metric(tsa_text, "execution")
    result = {
        "status": "passed",
        "record_count": len(sample_ids),
        "metrics_fraction": {"em": em, "ex": ex, "tsa": tsa},
        "metrics_percent": {
            "em": round(em * 100, 2),
            "ex": round(ex * 100, 2),
            "tsa": round(tsa * 100, 2),
        },
        "prediction_format": {
            "format_valid_count": sum(
                bool(item.get("format_valid")) for item in prediction_records
            ),
            "format_valid_percent": round(
                100
                * sum(bool(item.get("format_valid")) for item in prediction_records)
                / len(prediction_records),
                2,
            ),
        },
        "official_runs": {
            "spider_em_ex": original_result,
            "test_suite_tsa": tsa_result,
        },
        "official_source_sha256": source_hashes,
        "input_sha256": {
            "predictions_jsonl": sha256_file(paths["predictions"]),
            "pred_sql": sha256_file(official_pred),
            "gold_sql": sha256_file(paths["gold"]),
            "tables_json": sha256_file(paths["tables"]),
        },
        "value_policy": "model_predicts_values; --plug_value disabled",
        "distinct_policy": "official default; --keep_distinct disabled",
    }
    write_json(output / "metrics.json", result)
    return result
