"""V100 与 RTX 4090 的最小精度差异。"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from .evaluation_config import EvaluationConfig
from .training_config import TrainingConfig


HardwareName = Literal["v100", "4090"]


def apply_training_hardware(
    config: TrainingConfig,
    hardware: HardwareName | None,
) -> TrainingConfig:
    """在不复制 YAML 的情况下应用训练精度。"""

    if hardware is None:
        return config
    if hardware == "v100":
        return replace(config, bf16=False, fp16=True, tf32=False)
    if hardware == "4090":
        return replace(config, bf16=True, fp16=False, tf32=True)
    raise ValueError(f"不支持的硬件：{hardware}")


def apply_evaluation_hardware(
    config: EvaluationConfig,
    hardware: HardwareName | None,
) -> EvaluationConfig:
    """让推理计算精度与训练硬件保持一致。"""

    if hardware is None:
        return config
    if hardware == "v100":
        return replace(config, bf16=False)
    if hardware == "4090":
        return replace(config, bf16=True)
    raise ValueError(f"不支持的硬件：{hardware}")
