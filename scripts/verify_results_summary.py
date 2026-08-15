"""核对可公开结果摘要与本地官方评测原始指标。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = PROJECT_ROOT / "results_summary" / "official_metrics.json"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "results" / "evaluation"


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_equal(label: str, public: object, raw: object) -> None:
    if public != raw:
        raise ValueError(f"{label} 不一致：public={public!r}, raw={raw!r}")


def _locate_raw_root(raw_root: Path, names: list[str]) -> Path:
    """兼容当前扁平输出和早期按硬件归档的本地结果。"""

    for candidate in (raw_root, raw_root / "v100"):
        if all(
            (candidate / name / "official_metrics" / "metrics.json").is_file()
            for name in names
        ):
            return candidate
    raise FileNotFoundError(f"找不到完整原始指标目录：{raw_root}")


def verify(summary_path: Path, raw_root: Path) -> dict[str, object]:
    summary = _load_json(summary_path)
    benchmark = summary["benchmark"]
    source_hashes = summary["official_source_sha256"]
    checked: list[str] = []
    experiments = summary["experiments"]
    raw_root = _locate_raw_root(
        raw_root,
        [experiment["name"] for experiment in experiments],
    )

    for experiment in experiments:
        name = experiment["name"]
        raw = _load_json(raw_root / name / "official_metrics" / "metrics.json")
        _assert_equal(f"{name}.status", experiment["status"], raw["status"])
        _assert_equal(
            f"{name}.record_count",
            experiment["record_count"],
            raw["record_count"],
        )
        _assert_equal(
            f"{name}.metrics_percent",
            experiment["metrics_percent"],
            raw["metrics_percent"],
        )
        _assert_equal(
            f"{name}.format_valid_count",
            experiment["format_valid_count"],
            raw["prediction_format"]["format_valid_count"],
        )
        _assert_equal(
            f"{name}.format_valid_percent",
            experiment["format_valid_percent"],
            raw["prediction_format"]["format_valid_percent"],
        )
        _assert_equal(
            f"{name}.official_source_sha256",
            source_hashes,
            raw["official_source_sha256"],
        )
        _assert_equal(
            f"{name}.gold_sql_sha256",
            benchmark["gold_sql_sha256"],
            raw["input_sha256"]["gold_sql"],
        )
        _assert_equal(
            f"{name}.tables_json_sha256",
            benchmark["tables_json_sha256"],
            raw["input_sha256"]["tables_json"],
        )
        for key in ("predictions_jsonl", "pred_sql"):
            _assert_equal(
                f"{name}.{key}",
                experiment["input_sha256"][key],
                raw["input_sha256"][key],
            )
        checked.append(name)

    return {"status": "passed", "checked_experiments": checked}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(args.summary.resolve(), args.raw_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
