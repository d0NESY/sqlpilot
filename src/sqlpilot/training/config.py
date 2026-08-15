"""读取并严格验证 QLoRA YAML 配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    experiment_name: str
    model_name: str
    model_revision: str
    train_file: Path
    eval_file: Path
    output_dir: Path
    expected_train_records: int
    expected_eval_records: int
    expected_train_sha256: str
    expected_eval_sha256: str
    max_length: int = 4096
    num_train_epochs: float = 3.0
    max_steps: int = -1
    learning_rate: float = 2.0e-4
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    seed: int = 42
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    optim: str = "paged_adamw_8bit"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    dataset_num_proc: int = 4
    allow_truncation: bool = False
    resume_from_checkpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field in ("train_file", "eval_file", "output_dir"):
            value[field] = str(value[field])
        value["lora_target_modules"] = list(
            value["lora_target_modules"]
        )
        return value


def _resolve_path(project_root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空路径")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_training_config(
    config_path: str | Path,
    project_root: str | Path,
) -> TrainingConfig:
    """读取 YAML；未知字段会直接报错，避免拼写错误被忽略。"""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("训练配置最外层必须是对象")

    root = Path(project_root).resolve()
    raw["train_file"] = _resolve_path(root, raw.get("train_file"), "train_file")
    raw["eval_file"] = _resolve_path(root, raw.get("eval_file"), "eval_file")
    raw["output_dir"] = _resolve_path(
        root,
        raw.get("output_dir"),
        "output_dir",
    )
    if "lora_target_modules" in raw:
        modules = raw["lora_target_modules"]
        if not isinstance(modules, list) or not all(
            isinstance(module, str) and module for module in modules
        ):
            raise ValueError("lora_target_modules 必须是非空字符串列表")
        raw["lora_target_modules"] = tuple(modules)

    config = TrainingConfig(**raw)
    _validate_config(config)
    return config


def _validate_config(config: TrainingConfig) -> None:
    if not config.experiment_name.strip():
        raise ValueError("experiment_name 不能为空")
    if not config.model_name.strip():
        raise ValueError("model_name 不能为空")
    if (
        not config.model_revision.strip()
        or config.model_revision in {"main", "master", "latest"}
    ):
        raise ValueError("model_revision 必须固定到 commit，不能使用浮动分支")
    for path, field in (
        (config.train_file, "train_file"),
        (config.eval_file, "eval_file"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{field} 不存在：{path}")
    if "official_dev_evaluation_only" in config.eval_file.name:
        raise ValueError("官方 dev 不能作为训练过程中的 eval_file")
    if config.max_length <= 0:
        raise ValueError("max_length 必须大于 0")
    if config.expected_train_records <= 0 or config.expected_eval_records <= 0:
        raise ValueError("预期记录数必须大于 0")
    if config.bf16 == config.fp16:
        raise ValueError("bf16 和 fp16 必须且只能启用一个")
    if config.train_batch_size <= 0 or config.eval_batch_size <= 0:
        raise ValueError("batch size 必须大于 0")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps 必须大于 0")
    if min(config.logging_steps, config.eval_steps, config.save_steps) <= 0:
        raise ValueError("logging/eval/save steps 必须大于 0")
    if config.save_total_limit < 2:
        raise ValueError("save_total_limit 至少为 2，以同时保留最佳与最新 checkpoint")
    if not config.load_best_model_at_end:
        raise ValueError("正式训练必须在结束时自动回载最佳 checkpoint")
    if config.metric_for_best_model != "eval_loss":
        raise ValueError("本项目固定使用 eval_loss 选择最佳 checkpoint")
    if config.greater_is_better:
        raise ValueError("eval_loss 越小越好，greater_is_better 必须为 false")
    if config.save_steps % config.eval_steps != 0:
        raise ValueError("save_steps 必须是 eval_steps 的整数倍")
    if config.lora_r <= 0 or config.lora_alpha <= 0:
        raise ValueError("LoRA r 和 alpha 必须大于 0")
    if not 0 <= config.lora_dropout < 1:
        raise ValueError("lora_dropout 必须在 [0, 1) 内")
    for digest, field in (
        (config.expected_train_sha256, "expected_train_sha256"),
        (config.expected_eval_sha256, "expected_eval_sha256"),
    ):
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"{field} 必须是小写 SHA-256")
