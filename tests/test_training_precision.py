from __future__ import annotations

import torch

from sqlpilot.training.train_qlora import (
    _prepare_trainable_parameters_for_mixed_precision,
)


def test_fp16_profile_upcasts_only_trainable_parameters() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    ).to(dtype=torch.bfloat16)
    model[0].weight.requires_grad_(False)

    report = _prepare_trainable_parameters_for_mixed_precision(
        model,
        force_fp32=True,
    )

    assert model[0].weight.dtype == torch.bfloat16
    assert model[1].weight.dtype == torch.float32
    assert report["before"] == {
        "bfloat16": {"tensor_count": 1, "parameter_count": 8}
    }
    assert report["after"] == {
        "float32": {"tensor_count": 1, "parameter_count": 8}
    }
    assert report["converted_tensor_count"] == 1
    assert report["converted_parameter_count"] == 8


def test_bf16_profile_keeps_trainable_parameters_unchanged() -> None:
    model = torch.nn.Linear(4, 2, bias=False).to(dtype=torch.bfloat16)

    report = _prepare_trainable_parameters_for_mixed_precision(
        model,
        force_fp32=False,
    )

    assert model.weight.dtype == torch.bfloat16
    assert report["before"] == report["after"]
    assert report["converted_tensor_count"] == 0
    assert report["converted_parameter_count"] == 0
