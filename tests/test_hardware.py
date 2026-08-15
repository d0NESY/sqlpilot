"""单套配置在 V100 与 RTX 4090 间的精度切换测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlpilot.evaluation_config import load_evaluation_config
from sqlpilot.hardware import apply_evaluation_hardware, apply_training_hardware
from sqlpilot.training_config import load_training_config


class HardwareTestCase(unittest.TestCase):
    @staticmethod
    def _materialize_inputs(root: Path) -> None:
        datasets = root / "data" / "processed" / "datasets"
        for name in (
            "s1_ddl",
            "s2_structured",
            "s3_values",
            "s4_schema_link",
        ):
            directory = datasets / name
            directory.mkdir(parents=True, exist_ok=True)
            for filename in (
                "train.jsonl",
                "validation.jsonl",
                "official_dev_evaluation_only.jsonl",
            ):
                (directory / filename).write_text("", encoding="utf-8")

    def test_one_training_config_set_supports_both_gpus(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            test_root = Path(directory)
            self._materialize_inputs(test_root)
            for stage in ("s1", "s2", "s3", "s4"):
                base = load_training_config(
                    project_root / "configs" / "experiments" / f"{stage}.yaml",
                    test_root,
                )
                v100 = apply_training_hardware(base, "v100")
                rtx4090 = apply_training_hardware(base, "4090")
                self.assertEqual(
                    (v100.bf16, v100.fp16, v100.tf32),
                    (False, True, False),
                )
                self.assertEqual(
                    (rtx4090.bf16, rtx4090.fp16, rtx4090.tf32),
                    (True, False, True),
                )
                self.assertNotIn("v100", base.output_dir.as_posix().lower())
                self.assertNotIn("modern", base.output_dir.as_posix().lower())

    def test_one_evaluation_config_set_supports_both_gpus(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        names = (
            "base_s4",
            "s1_adapter",
            "s2_adapter",
            "s3_adapter",
            "s4_adapter",
        )
        with tempfile.TemporaryDirectory() as directory:
            test_root = Path(directory)
            self._materialize_inputs(test_root)
            for name in names:
                base = load_evaluation_config(
                    project_root / "configs" / "evaluation" / f"{name}.yaml",
                    test_root,
                )
                self.assertFalse(apply_evaluation_hardware(base, "v100").bf16)
                self.assertTrue(apply_evaluation_hardware(base, "4090").bf16)
                self.assertNotIn("v100", base.output_dir.as_posix().lower())
                self.assertNotIn("modern", base.output_dir.as_posix().lower())


if __name__ == "__main__":
    unittest.main()
