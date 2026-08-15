"""训练配置、数据身份和 Token 安全预检测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sqlpilot.training.config import load_training_config
from sqlpilot.training.preflight import (
    analyze_jsonl,
    enforce_length_safety,
    validate_dataset_identity,
)


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    ):
        del tokenize
        tokens = [
            token
            for message in messages
            for token in message["content"].split()
        ]
        ids = list(range(len(tokens)))
        if add_generation_prompt:
            ids.append(98)
        else:
            ids.append(self.eos_token_id)
        if return_dict:
            return {"input_ids": ids}
        return ids


class TrainingPreflightTestCase(unittest.TestCase):
    def test_real_s2_config_is_valid_and_uses_internal_validation(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        train_file = (
            project_root
            / "data"
            / "processed"
            / "datasets"
            / "s2_structured"
            / "train.jsonl"
        )
        if not train_file.is_file():
            self.skipTest("需要本地生成的 S2 训练与内部验证数据")
        config = load_training_config(
            project_root / "configs" / "experiments" / "s2.yaml",
            project_root,
        )

        self.assertEqual(
            config.model_revision,
            "488639f1ff808d1d3d0ba301aef8c11461451ec5",
        )
        self.assertIn("s2_structured", str(config.train_file))
        self.assertNotIn(
            "official_dev_evaluation_only",
            config.eval_file.name,
        )
        self.assertTrue(config.load_best_model_at_end)
        self.assertEqual(config.metric_for_best_model, "eval_loss")
        self.assertFalse(config.greater_is_better)
        self.assertEqual(config.save_steps % config.eval_steps, 0)
        self.assertFalse(config.bf16)
        self.assertTrue(config.fp16)
        self.assertFalse(config.tf32)
        train_identity = validate_dataset_identity(
            config.train_file,
            config.expected_train_records,
            config.expected_train_sha256,
        )
        self.assertEqual(train_identity["record_count"], 7939)

    def test_token_report_detects_truncation(self) -> None:
        record = {
            "prompt": [
                {"role": "system", "content": "one two"},
                {
                    "role": "user",
                    "content": "three four five six",
                },
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": "seven eight nine",
                }
            ],
            "db_id": "demo",
            "sample_id": "demo_1",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )
            report = analyze_jsonl(
                path,
                FakeTokenizer(),
                max_length=5,
            )

        self.assertEqual(report["record_count"], 1)
        self.assertEqual(report["over_max_count"], 1)
        self.assertEqual(report["eos_missing_count"], 0)
        with self.assertRaises(ValueError):
            enforce_length_safety(report, allow_truncation=False)


if __name__ == "__main__":
    unittest.main()
