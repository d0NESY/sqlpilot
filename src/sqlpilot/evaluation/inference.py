"""在固定基础模型上执行可恢复的 baseline/Adapter 贪心推理。"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from sqlpilot.data import sha256_file, write_json
from sqlpilot.hub import resolve_model_source

from .config import EvaluationConfig
from .prediction import extract_sql, load_prediction_records


TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "huggingface-hub",
    "peft",
    "bitsandbytes",
)


def _command_output(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": f"{type(error).__name__}: {error}"}


def _environment() -> dict[str, Any]:
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
    }


def _load_dataset(config: EvaluationConfig) -> list[dict[str, Any]]:
    actual_hash = sha256_file(config.dataset_file)
    if actual_hash != config.expected_dataset_sha256:
        raise ValueError(
            "评估数据 SHA-256 不一致："
            f"expected={config.expected_dataset_sha256}, actual={actual_hash}"
        )
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with config.dataset_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id")
            prompt = record.get("prompt")
            if not isinstance(sample_id, str) or sample_id in seen_ids:
                raise ValueError(f"第 {line_number} 行 sample_id 非法或重复")
            if (
                not isinstance(prompt, list)
                or [item.get("role") for item in prompt]
                != ["system", "user"]
            ):
                raise ValueError(f"{sample_id} prompt 格式错误")
            seen_ids.add(sample_id)
            records.append(record)
    if len(records) != config.expected_records:
        raise ValueError(
            f"评估记录数不一致：expected={config.expected_records}, "
            f"actual={len(records)}"
        )
    return records


def _chat_feature(tokenizer: Any, prompt: list[dict[str, str]]) -> dict[str, Any]:
    try:
        encoded = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
        )
        encoded = {"input_ids": ids, "attention_mask": [1] * len(ids)}
    if not isinstance(encoded, Mapping):
        encoded = {"input_ids": encoded}
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    attention = encoded.get("attention_mask", [1] * len(input_ids))
    if attention and isinstance(attention[0], list):
        attention = attention[0]
    return {"input_ids": list(input_ids), "attention_mask": list(attention)}


def run_predictions(
    config: EvaluationConfig,
    resume: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """逐批生成并每条落盘；中断后可用 --resume 从严格前缀继续。"""

    if limit is not None and limit <= 0:
        raise ValueError("limit 必须大于 0")
    records = _load_dataset(config)
    target_count = len(records) if limit is None else min(limit, len(records))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = config.output_dir / "predictions.jsonl"
    resolved_path = config.output_dir / "resolved_config.json"
    environment_path = config.output_dir / "environment.json"
    resolved_config = config.to_dict()

    existing: list[dict[str, Any]] = []
    if prediction_path.exists():
        if not resume:
            raise FileExistsError(
                f"预测文件已存在；确认配置后使用 --resume：{prediction_path}"
            )
        existing = load_prediction_records(prediction_path)
        expected_prefix = [item["sample_id"] for item in records[: len(existing)]]
        if [item["sample_id"] for item in existing] != expected_prefix:
            raise ValueError("已有预测不是当前数据集的严格前缀，拒绝续跑")
        if not resolved_path.is_file():
            raise FileNotFoundError("已有预测缺少 resolved_config.json，拒绝续跑")
        previous_config = json.loads(resolved_path.read_text(encoding="utf-8"))
        if previous_config != resolved_config:
            raise ValueError("当前配置与已有预测的 resolved_config.json 不一致")
    if len(existing) >= target_count:
        return {
            "status": "already_complete",
            "prediction_path": str(prediction_path),
            "record_count": len(existing),
        }

    write_json(resolved_path, resolved_config)
    write_json(environment_path, _environment())

    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError("推理依赖未安装；请按 requirements.txt 安装") from error

    if not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA GPU，拒绝启动正式推理")
    if config.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持配置要求的 BF16")
    if config.adapter_path is not None:
        if not config.adapter_path.is_dir():
            raise FileNotFoundError(f"Adapter 目录不存在：{config.adapter_path}")
        if not (config.adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError("Adapter 目录缺少 adapter_config.json")

    model_source = resolve_model_source(
        config.model_name,
        config.model_revision,
    )
    resolved_revision = model_source.resolved_revision

    set_seed(config.seed)
    compute_dtype = torch.bfloat16 if config.bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        model_source.source,
        revision=None if model_source.offline else resolved_revision,
        local_files_only=model_source.offline,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source.source,
        revision=None if model_source.offline else resolved_revision,
        local_files_only=model_source.offline,
        dtype=compute_dtype,
        quantization_config=quantization,
        device_map={"": 0},
    )
    if config.adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(config.adapter_path))
    model.eval()

    start_index = len(existing)
    started = time.monotonic()
    with prediction_path.open("a", encoding="utf-8", newline="\n") as output:
        for batch_start in range(start_index, target_count, config.batch_size):
            batch_records = records[
                batch_start : min(batch_start + config.batch_size, target_count)
            ]
            features = [
                _chat_feature(tokenizer, item["prompt"]) for item in batch_records
            ]
            for item, feature in zip(batch_records, features):
                if len(feature["input_ids"]) > config.max_input_length:
                    raise ValueError(
                        f"{item['sample_id']} prompt 长度 "
                        f"{len(feature['input_ids'])} 超过 "
                        f"{config.max_input_length}；禁止截断"
                    )
            encoded = tokenizer.pad(
                features,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            input_width = encoded["input_ids"].shape[1]
            batch_started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=config.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.monotonic() - batch_started
            decoded = tokenizer.batch_decode(
                generated[:, input_width:],
                skip_special_tokens=True,
            )
            for offset, (item, feature, token_ids, raw_output) in enumerate(zip(
                batch_records,
                features,
                generated[:, input_width:],
                decoded,
            )):
                extracted = extract_sql(raw_output, config.output_mode)
                schema_link_present = "<SCHEMA_LINK>" in raw_output.upper()
                prediction = {
                    "index": batch_start + offset,
                    "sample_id": item["sample_id"],
                    "db_id": item["db_id"],
                    "raw_output": raw_output,
                    "predicted_sql": extracted.sql,
                    "extraction_method": extracted.method,
                    "format_valid": extracted.format_valid and (
                        config.output_mode == "direct" or schema_link_present
                    ),
                    "schema_link_present": schema_link_present,
                    "prompt_tokens": len(feature["input_ids"]),
                    "generated_tokens": int(
                        token_ids.ne(tokenizer.pad_token_id).sum().item()
                    ),
                    "batch_generation_seconds": elapsed,
                }
                output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                output.flush()
            completed = min(batch_start + len(batch_records), target_count)
            total_elapsed = time.monotonic() - started
            rate = (completed - start_index) / max(total_elapsed, 1e-9)
            remaining = (target_count - completed) / max(rate, 1e-9)
            print(
                f"generated={completed}/{target_count} "
                f"rate={rate:.3f} samples/s eta={remaining / 60:.1f} min",
                flush=True,
            )

    final_records = load_prediction_records(prediction_path)
    summary = {
        "status": "complete" if len(final_records) == len(records) else "partial",
        "model_kind": "baseline" if config.is_baseline else "adapter",
        "resolved_model_revision": resolved_revision,
        "hf_hub_offline": model_source.offline,
        "prediction_path": str(prediction_path),
        "record_count": len(final_records),
        "expected_records": len(records),
        "format_valid_count": sum(
            bool(item.get("format_valid")) for item in final_records
        ),
        "invalid_sql_count": sum(
            item["predicted_sql"].startswith(
                "SELECT * FROM __sqlpilot_invalid_prediction__"
            )
            for item in final_records
        ),
    }
    write_json(config.output_dir / "prediction_summary.json", summary)
    return summary
