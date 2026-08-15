"""Prompt-Completion 训练数据构造测试。"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from sqlpilot.data import (
    SpiderTrainingDataBuilder,
    load_spider_samples,
    normalize_gold_sql,
    parse_spider_schema_entry,
    write_jsonl,
)


class TrainingDataTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        database_directory = self.root / "database" / "demo"
        database_directory.mkdir(parents=True)
        database_path = database_directory / "demo.sqlite"

        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                """
                CREATE TABLE singer (
                    singer_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                );
                """
            )
            connection.executemany(
                "INSERT INTO singer (singer_id, name) VALUES (?, ?);",
                [(1, "Alice"), (2, "Zoë")],
            )
            connection.commit()

        self.samples = [
            {
                "db_id": "demo",
                "question": "How many singers are there?",
                "query": "SELECT COUNT(*) FROM singer",
            },
            {
                "db_id": "demo",
                "question": "List all singer names.",
                "query": "SELECT name FROM singer;",
            },
        ]
        self.source_path = self.root / "train_spider.json"
        self.source_path.write_text(
            json.dumps(self.samples),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_builds_prompt_completion_and_reuses_schema(self) -> None:
        builder = SpiderTrainingDataBuilder(
            self.root / "database",
            schema_style="structured",
        )
        records = list(
            builder.build_records(self.samples, split="train")
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["sample_id"], "train_000001")
        self.assertEqual(records[0]["db_id"], "demo")
        self.assertIn(
            "Table: singer",
            records[0]["prompt"][1]["content"],
        )
        self.assertIn(
            "How many singers are there?",
            records[0]["prompt"][1]["content"],
        )
        self.assertEqual(
            records[0]["completion"][0]["content"],
            "SELECT COUNT(*) FROM singer;",
        )
        self.assertEqual(builder.cached_database_count, 1)

    def test_load_and_write_jsonl(self) -> None:
        loaded = load_spider_samples(self.source_path)
        builder = SpiderTrainingDataBuilder(self.root / "database")
        output_path = self.root / "processed" / "train.jsonl"

        count = write_jsonl(
            builder.build_records(loaded, split="train", limit=1),
            output_path,
        )
        output_lines = output_path.read_text(
            encoding="utf-8"
        ).splitlines()

        self.assertEqual(count, 1)
        self.assertEqual(len(output_lines), 1)
        self.assertEqual(
            json.loads(output_lines[0])["sample_id"],
            "train_000001",
        )

    def test_normalize_gold_sql(self) -> None:
        self.assertEqual(
            normalize_gold_sql(" SELECT 1;;;  "),
            "SELECT 1;",
        )
        with self.assertRaises(ValueError):
            normalize_gold_sql("  ")

    def test_rejects_invalid_limit_and_missing_fields(self) -> None:
        builder = SpiderTrainingDataBuilder(self.root / "database")

        with self.assertRaises(ValueError):
            list(
                builder.build_records(
                    self.samples,
                    split="train",
                    limit=-1,
                )
            )
        with self.assertRaises(ValueError):
            builder.build_record(
                {"db_id": "demo", "question": "Missing SQL"},
                sample_id="train_000003",
            )

    def test_builds_schema_link_completion_from_official_ast(self) -> None:
        official = parse_spider_schema_entry(
            {
                "db_id": "demo",
                "table_names_original": ["singer"],
                "column_names_original": [
                    [-1, "*"],
                    [0, "singer_id"],
                    [0, "name"],
                ],
                "column_types": ["text", "number", "text"],
                "primary_keys": [1],
                "foreign_keys": [],
            }
        )
        sample = {
            **self.samples[0],
            "sql": {
                "select": [
                    False,
                    [[3, [0, [0, 0, False], None]]],
                ],
                "from": {
                    "table_units": [["table_unit", 0]],
                    "conds": [],
                },
                "where": [],
                "groupBy": [],
                "having": [],
                "orderBy": [],
                "limit": None,
                "intersect": None,
                "union": None,
                "except": None,
            },
        }
        builder = SpiderTrainingDataBuilder(
            self.root / "database",
            schema_style="structured_with_values",
            official_schema_catalog={"demo": official},
            value_cache_root=self.root / "cache",
            output_mode="schema_link",
        )
        record = builder.build_record(sample, "train_000001")
        completion = record["completion"][0]["content"]

        self.assertIn("tables: singer", completion)
        self.assertIn("columns: none", completion)
        self.assertIn("<SQL>", completion)
        self.assertIn("SELECT COUNT(*) FROM singer;", completion)
        self.assertIn("examples=", record["prompt"][1]["content"])


if __name__ == "__main__":
    unittest.main()
