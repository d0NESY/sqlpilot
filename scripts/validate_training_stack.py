"""不下载模型，验证锁定训练库的版本和关键 API。"""

from __future__ import annotations

import inspect
import json
from importlib import metadata


EXPECTED_VERSIONS = {
    "transformers": "5.14.1",
    "datasets": "5.0.0",
    "accelerate": "1.14.0",
    "peft": "0.19.1",
    "trl": "1.8.0",
    "bitsandbytes": "0.50.0",
}


def main() -> None:
    import torch
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    actual_versions = {
        package: metadata.version(package)
        for package in EXPECTED_VERSIONS
    }
    mismatches = {
        package: {
            "expected": expected,
            "actual": actual_versions[package],
        }
        for package, expected in EXPECTED_VERSIONS.items()
        if actual_versions[package] != expected
    }
    if mismatches:
        raise RuntimeError(f"训练库版本不匹配：{mismatches}")

    sft_config_parameters = inspect.signature(SFTConfig).parameters
    required_config_parameters = {
        "max_length",
        "completion_only_loss",
        "eval_strategy",
        "use_cache",
        "model_init_kwargs",
        "truncation_mode",
        "shuffle_dataset",
    }
    missing_config = sorted(
        required_config_parameters - set(sft_config_parameters)
    )
    trainer_parameters = inspect.signature(SFTTrainer).parameters
    required_trainer_parameters = {
        "processing_class",
        "quantization_config",
        "peft_config",
    }
    missing_trainer = sorted(
        required_trainer_parameters - set(trainer_parameters)
    )
    if missing_config or missing_trainer:
        raise RuntimeError(
            "训练 API 不匹配："
            f"SFTConfig 缺少 {missing_config}；"
            f"SFTTrainer 缺少 {missing_trainer}"
        )

    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    SFTConfig(
        output_dir=".api_validation_only",
        max_length=4096,
        completion_only_loss=True,
        eval_strategy="steps",
        use_cache=False,
        bf16=False,
        fp16=False,
        report_to="none",
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "versions": actual_versions,
                "sft_config_parameters_checked": sorted(
                    required_config_parameters
                ),
                "sft_trainer_parameters_checked": sorted(
                    required_trainer_parameters
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
