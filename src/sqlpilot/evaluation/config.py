"""读取并严格验证 baseline/Adapter 推理配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_OUTPUT_MODES = {"direct", "schema_link"}


@dataclass(frozen=True)
class EvaluationConfig:
    experiment_name: str
    model_name: str
    model_revision: str
    dataset_file: Path
    output_dir: Path
    expected_records: int
    expected_dataset_sha256: str
    output_mode: str
    adapter_path: Path | None = None
    max_input_length: int = 4096
    max_new_tokens: int = 512
    batch_size: int = 1
    load_in_4bit: bool = True
    bf16: bool = True
    seed: int = 42

    @property
    def is_baseline(self) -> bool:
        return self.adapter_path is None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field in ("dataset_file", "output_dir", "adapter_path"):
            item = value[field]
            value[field] = None if item is None else str(item)
        value["is_baseline"] = self.is_baseline
        return value


def _resolve_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空路径")
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_evaluation_config(
    config_path: str | Path,
    project_root: str | Path,
) -> EvaluationConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("评估配置最外层必须是对象")

    root = Path(project_root).resolve()
    raw["dataset_file"] = _resolve_path(
        root, raw.get("dataset_file"), "dataset_file"
    )
    raw["output_dir"] = _resolve_path(
        root, raw.get("output_dir"), "output_dir"
    )
    if raw.get("adapter_path") is not None:
        raw["adapter_path"] = _resolve_path(
            root, raw["adapter_path"], "adapter_path"
        )

    config = EvaluationConfig(**raw)
    _validate_config(config)
    return config


def _validate_config(config: EvaluationConfig) -> None:
    if not config.experiment_name.strip():
        raise ValueError("experiment_name 不能为空")
    if not config.model_name.strip():
        raise ValueError("model_name 不能为空")
    if (
        not config.model_revision.strip()
        or config.model_revision in {"main", "master", "latest"}
    ):
        raise ValueError("model_revision 必须固定到 commit")
    if not config.dataset_file.is_file():
        raise FileNotFoundError(f"评估数据不存在：{config.dataset_file}")
    if "official_dev_evaluation_only" not in config.dataset_file.name:
        raise ValueError("正式 Spider 评估必须使用 evaluation-only 数据文件")
    if config.output_mode not in SUPPORTED_OUTPUT_MODES:
        raise ValueError(
            f"output_mode 必须是 {sorted(SUPPORTED_OUTPUT_MODES)}"
        )
    if config.expected_records <= 0:
        raise ValueError("expected_records 必须大于 0")
    digest = config.expected_dataset_sha256
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("expected_dataset_sha256 必须是小写 SHA-256")
    if config.max_input_length <= 0 or config.max_new_tokens <= 0:
        raise ValueError("Token 长度必须大于 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if not config.load_in_4bit:
        raise ValueError("本项目正式评估固定使用与训练一致的 4-bit 权重")
