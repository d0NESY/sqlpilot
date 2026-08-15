"""零样本/Adapter 评估配置、SQL 提取和官方顺序测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sqlpilot.evaluation_config import load_evaluation_config
from sqlpilot.official_evaluation import _overall_metric
from sqlpilot.prediction import (
    INVALID_SQL,
    extract_sql,
    materialize_official_predictions,
)
from sqlpilot.spider_evaluation import prepare_spider_dev_evaluation


class EvaluationTestCase(unittest.TestCase):
    @staticmethod
    def _materialize_evaluation_inputs(root: Path) -> None:
        datasets = root / "data" / "processed" / "datasets"
        for name in (
            "s1_ddl",
            "s2_structured",
            "s3_values",
            "s4_schema_link",
        ):
            path = datasets / name / "official_dev_evaluation_only.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def test_real_baseline_config_is_pinned_and_has_no_adapter(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            test_root = Path(directory)
            self._materialize_evaluation_inputs(test_root)
            config = load_evaluation_config(
                project_root / "configs" / "evaluation" / "base_s4.yaml",
                test_root,
            )
        self.assertTrue(config.is_baseline)
        self.assertEqual(config.expected_records, 1034)
        self.assertEqual(config.output_mode, "schema_link")
        self.assertFalse(config.bf16)
        self.assertEqual(
            config.model_revision,
            "488639f1ff808d1d3d0ba301aef8c11461451ec5",
        )

    def test_all_real_evaluation_configs_are_valid(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config_dir = project_root / "configs" / "evaluation"
        with tempfile.TemporaryDirectory() as directory:
            test_root = Path(directory)
            self._materialize_evaluation_inputs(test_root)
            configs = [
                load_evaluation_config(path, test_root)
                for path in sorted(config_dir.glob("*.yaml"))
            ]
        self.assertEqual(len(configs), 6)
        self.assertEqual(sum(item.is_baseline for item in configs), 2)
        self.assertEqual(sum(not item.is_baseline for item in configs), 4)
        self.assertTrue(all(item.expected_records == 1034 for item in configs))
        self.assertTrue(all(not item.bf16 for item in configs))

    def test_extract_sql_uses_fixed_priority_and_rejects_writes(self) -> None:
        tagged = extract_sql(
            "<SCHEMA_LINK>tables: singer</SCHEMA_LINK>\n"
            "<SQL>SELECT count(*)\nFROM singer;</SQL>",
            "schema_link",
        )
        self.assertEqual(tagged.sql, "SELECT count(*) FROM singer;")
        self.assertEqual(tagged.method, "sql_tag")
        self.assertTrue(tagged.format_valid)

        fenced = extract_sql("```sql\nSELECT ';' AS value;\n```", "direct")
        self.assertEqual(fenced.sql, "SELECT ';' AS value;")
        self.assertTrue(fenced.format_valid)

        unsafe = extract_sql(
            "WITH chosen AS (SELECT 1) DELETE FROM singer;",
            "direct",
        )
        self.assertEqual(unsafe.sql, INVALID_SQL)
        self.assertFalse(unsafe.format_valid)

    def test_official_prediction_materialization_requires_exact_order(self) -> None:
        records = [
            {"sample_id": "dev_000001", "predicted_sql": "SELECT 1;"},
            {"sample_id": "dev_000002", "predicted_sql": "SELECT 2;"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pred.sql"
            materialize_official_predictions(
                records,
                ["dev_000001", "dev_000002"],
                output,
            )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "SELECT 1;\nSELECT 2;\n",
            )
            with self.assertRaises(ValueError):
                materialize_official_predictions(
                    list(reversed(records)),
                    ["dev_000001", "dev_000002"],
                    output,
                )

    def test_official_metric_parser_keeps_full_precision(self) -> None:
        output = (
            "execution          0.418762088975\n"
            "exact match        0.326885880078\n"
        )
        self.assertEqual(_overall_metric(output, "execution"), 0.418762088975)
        self.assertEqual(
            _overall_metric(output, "exact match"), 0.326885880078
        )

    def test_real_spider_dev_variants_produce_one_gold_order(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        datasets = project_root / "data" / "processed" / "datasets"
        raw_dev = project_root / "data" / "raw" / "spider_data" / "dev.json"
        required_variants = tuple(
            datasets / name / "official_dev_evaluation_only.jsonl"
            for name in (
                "s1_ddl",
                "s2_structured",
                "s3_values",
                "s4_schema_link",
            )
        )
        if not raw_dev.is_file() or not all(
            path.is_file() for path in required_variants
        ):
            self.skipTest("需要本地 Spider dev 和四组派生评测数据")
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare_spider_dev_evaluation(
                raw_dev,
                {
                    name: path
                    for name, path in zip(
                        (
                            "s1_ddl",
                            "s2_structured",
                            "s3_values",
                            "s4_schema_link",
                        ),
                        required_variants,
                        strict=True,
                    )
                },
                directory,
            )
            lines = (Path(directory) / "gold.sql").read_text(
                encoding="utf-8"
            ).splitlines()
        self.assertEqual(manifest["record_count"], 1034)
        self.assertEqual(manifest["database_count"], 20)
        self.assertEqual(manifest["raw_dev"]["path"], "data/raw/spider_data/dev.json")
        self.assertEqual(len(lines), 1034)
        self.assertTrue(lines[0].endswith("\tconcert_singer"))


if __name__ == "__main__":
    unittest.main()
