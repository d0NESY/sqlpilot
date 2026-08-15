"""V100 与现代 GPU profile 配置隔离和精度测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlpilot.evaluation.config import load_evaluation_config
from sqlpilot.training.config import load_training_config


class HardwareProfileTestCase(unittest.TestCase):
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

    def test_training_profiles_are_complete_and_isolated(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            test_root = Path(directory)
            self._materialize_inputs(test_root)
            for profile, precision in {
                "v100": (False, True, False),
                "modern": (True, False, True),
            }.items():
                for stage in ("s1", "s2", "s3", "s4"):
                    config = load_training_config(
                        root
                        / "configs"
                        / profile
                        / "experiments"
                        / f"{stage}.yaml",
                        test_root,
                    )
                    self.assertEqual(
                        (config.bf16, config.fp16, config.tf32), precision
                    )
                    self.assertIn(
                        f"checkpoints/{profile}/", config.output_dir.as_posix()
                    )

    def test_evaluation_profiles_match_training_precision(self) -> None:
        root = Path(__file__).resolve().parents[1]
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
            for profile, bf16 in {"v100": False, "modern": True}.items():
                for name in names:
                    config = load_evaluation_config(
                        root
                        / "configs"
                        / profile
                        / "evaluation"
                        / f"{name}.yaml",
                        test_root,
                    )
                    self.assertEqual(config.bf16, bf16)
                    self.assertIn(
                        f"results/evaluation/{profile}/",
                        config.output_dir.as_posix(),
                    )
                    if not config.is_baseline:
                        self.assertIn(
                            f"checkpoints/{profile}/",
                            config.adapter_path.as_posix(),
                        )


if __name__ == "__main__":
    unittest.main()
