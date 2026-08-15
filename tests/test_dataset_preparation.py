"""全量数据准备中的排除和数据库级切分测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sqlpilot.dataset_preparation import (
    IndexedSpiderSample,
    load_indexed_samples,
    split_by_database,
)
from sqlpilot.data_validation import sha256_text


class DatasetPreparationTestCase(unittest.TestCase):
    def test_exclusion_requires_exact_query_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            samples = [
                {
                    "db_id": "db_a",
                    "question": "Q1",
                    "query": "SELECT 1",
                },
                {
                    "db_id": "db_b",
                    "question": "Q2",
                    "query": "SELECT 2",
                },
            ]
            path.write_text(json.dumps(samples), encoding="utf-8")
            exclusions = {
                ("train_spider", 1): {
                    "source": "train_spider",
                    "index": 1,
                    "db_id": "db_b",
                    "query_sha256": sha256_text("SELECT 2"),
                    "reason": "demo",
                }
            }

            loaded, excluded = load_indexed_samples(
                {"train_spider": path},
                exclusions,
            )

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].sample_id, "train_spider_000001")
            self.assertEqual(len(excluded), 1)

    def test_database_split_is_reproducible_and_disjoint(self) -> None:
        samples = [
            IndexedSpiderSample(
                source="train_spider",
                index=index,
                sample={
                    "db_id": f"db_{index % 10}",
                    "question": "Q",
                    "query": "SELECT 1",
                },
            )
            for index in range(100)
        ]

        train_a, validation_a, databases_a = split_by_database(
            samples,
            validation_ratio=0.2,
            seed=42,
        )
        train_b, validation_b, databases_b = split_by_database(
            samples,
            validation_ratio=0.2,
            seed=42,
        )

        self.assertEqual(databases_a, databases_b)
        self.assertEqual(
            [sample.sample_id for sample in train_a],
            [sample.sample_id for sample in train_b],
        )
        self.assertEqual(
            {sample.db_id for sample in train_a}
            & {sample.db_id for sample in validation_a},
            set(),
        )
        self.assertEqual(len(validation_a), 20)
        self.assertEqual(len(validation_b), 20)


if __name__ == "__main__":
    unittest.main()
