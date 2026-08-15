"""基于当前 TRL SFTTrainer API 的单卡 QLoRA 训练入口。"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .data_validation import write_json
from .hub import resolve_model_source
from .preflight import (
    analyze_jsonl,
    enforce_length_safety,
    validate_dataset_identity,
)
from .training_config import TrainingConfig


TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "bitsandbytes",
    "huggingface-hub",
)


def _command_output(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": f"{type(error).__name__}: {error}"}


def capture_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "NOT_INSTALLED"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git": _command_output(["git", "rev-parse", "HEAD"]),
        "nvidia_smi": _command_output(["nvidia-smi"]),
        "pip_freeze": _command_output(
            [sys.executable, "-m", "pip", "freeze"]
        ),
    }


def _trainable_dtype_counts(model: Any) -> dict[str, dict[str, int]]:
    """Count trainable tensors and parameters by dtype."""

    counts: dict[str, dict[str, int]] = {}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        dtype_name = str(parameter.dtype).removeprefix("torch.")
        entry = counts.setdefault(
            dtype_name,
            {"tensor_count": 0, "parameter_count": 0},
        )
        entry["tensor_count"] += 1
        entry["parameter_count"] += parameter.numel()
    return counts


def _prepare_trainable_parameters_for_mixed_precision(
    model: Any,
    *,
    force_fp32: bool,
) -> dict[str, Any]:
    """Keep FP16 GradScaler away from BF16 LoRA gradients.

    PEFT adapters are small compared with the frozen 4-bit base model.  Keeping
    only trainable adapter parameters in FP32 is both stable and inexpensive;
    the frozen quantized base model and its compute dtype are left untouched.
    """

    before = _trainable_dtype_counts(model)
    converted_tensor_count = 0
    converted_parameter_count = 0
    if force_fp32:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if not parameter.is_floating_point():
                raise TypeError(
                    f"Trainable parameter {name!r} is not floating point: "
                    f"{parameter.dtype}"
                )
            if str(parameter.dtype) != "torch.float32":
                parameter.data = parameter.data.float()
                converted_tensor_count += 1
                converted_parameter_count += parameter.numel()

    after = _trainable_dtype_counts(model)
    if force_fp32 and set(after) != {"float32"}:
        raise RuntimeError(
            "FP16 training requires every trainable adapter parameter to be "
            f"FP32, but found: {sorted(after)}"
        )
    return {
        "force_fp32": force_fp32,
        "before": before,
        "after": after,
        "converted_tensor_count": converted_tensor_count,
        "converted_parameter_count": converted_parameter_count,
    }


def run_training(
    config: TrainingConfig,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """先完成身份与 Token 预检，再选择停止或启动 QLoRA。"""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "预检依赖未安装；请先按 README 安装服务器环境"
        ) from error

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "resolved_config.json", config.to_dict())
    write_json(config.output_dir / "environment.json", capture_environment())

    train_identity = validate_dataset_identity(
        config.train_file,
        config.expected_train_records,
        config.expected_train_sha256,
    )
    eval_identity = validate_dataset_identity(
        config.eval_file,
        config.expected_eval_records,
        config.expected_eval_sha256,
    )

    model_source = resolve_model_source(
        config.model_name,
        config.model_revision,
    )
    resolved_revision = model_source.resolved_revision
    tokenizer = AutoTokenizer.from_pretrained(
        model_source.source,
        revision=None if model_source.offline else resolved_revision,
        local_files_only=model_source.offline,
    )
    if tokenizer.chat_template is None:
        raise ValueError("模型 tokenizer 没有 chat_template")
    if tokenizer.eos_token_id is None:
        raise ValueError("模型 tokenizer 没有 EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_lengths = analyze_jsonl(
        config.train_file,
        tokenizer,
        config.max_length,
    )
    eval_lengths = analyze_jsonl(
        config.eval_file,
        tokenizer,
        config.max_length,
    )
    report = {
        "status": "checking",
        "resolved_model_revision": resolved_revision,
        "model_license": model_source.model_license,
        "hf_hub_offline": model_source.offline,
        "train_identity": train_identity,
        "eval_identity": eval_identity,
        "train_token_lengths": train_lengths,
        "eval_token_lengths": eval_lengths,
        "completion_only_loss": True,
        "official_dev_used_for_training_eval": False,
    }
    try:
        enforce_length_safety(train_lengths, config.allow_truncation)
        enforce_length_safety(eval_lengths, config.allow_truncation)
    except ValueError as error:
        report["status"] = "failed"
        report["error"] = str(error)
        write_json(config.output_dir / "preflight_report.json", report)
        raise
    report["status"] = "passed"
    write_json(config.output_dir / "preflight_report.json", report)
    if preflight_only:
        return report

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError(
            "QLoRA 训练依赖未安装；请按 README 安装完整服务器环境"
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA GPU，已拒绝启动 QLoRA")
    if config.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "当前 GPU 不支持 BF16；请改用显式的 FP16 配置后重新预检"
        )

    compute_dtype = torch.bfloat16 if config.bf16 else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.lora_target_modules),
    )
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(config.train_file),
            "validation": str(config.eval_file),
        },
    )
    model_init_kwargs = {
        "dtype": compute_dtype,
        "use_cache": False,
        "local_files_only": model_source.offline,
    }
    if not model_source.offline:
        model_init_kwargs["revision"] = resolved_revision

    training_args = SFTConfig(
        output_dir=str(config.output_dir),
        run_name=config.experiment_name,
        model_init_kwargs=model_init_kwargs,
        max_length=config.max_length,
        completion_only_loss=True,
        packing=False,
        truncation_mode="keep_start",
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=config.bf16,
        fp16=config.fp16,
        tf32=config.tf32,
        optim=config.optim,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        report_to=["tensorboard"],
        seed=config.seed,
        data_seed=config.seed,
        dataset_num_proc=config.dataset_num_proc,
        shuffle_dataset=True,
        use_cache=False,
    )
    trainer = SFTTrainer(
        model=model_source.source,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        quantization_config=quantization_config,
        peft_config=lora_config,
    )
    trainable_dtype_report = _prepare_trainable_parameters_for_mixed_precision(
        trainer.model,
        force_fp32=config.fp16,
    )
    write_json(
        config.output_dir / "trainable_dtype_report.json",
        trainable_dtype_report,
    )
    print(
        "trainable dtype report: "
        + json.dumps(trainable_dtype_report, ensure_ascii=False)
    )
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    result = trainer.train(
        resume_from_checkpoint=config.resume_from_checkpoint
    )
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    metrics = dict(result.metrics)
    metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint
    metrics["best_metric"] = trainer.state.best_metric
    metrics["saved_model_is_best_checkpoint"] = bool(
        config.load_best_model_at_end and trainer.state.best_model_checkpoint
    )
    write_json(config.output_dir / "train_metrics.json", metrics)
    return {"preflight": report, "train_metrics": metrics}
