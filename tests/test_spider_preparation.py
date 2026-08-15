"""Spider 官方 Schema、示例值和 Schema Link 测试。"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from sqlpilot.schema_linker import extract_schema_link, serialize_schema_link
from sqlpilot.schema_parser import parse_sqlite_schema
from sqlpilot.schema_serializer import serialize_schema
from sqlpilot.spider_schema import (
    compare_spider_and_sqlite_schema,
    parse_spider_schema_entry,
)
from sqlpilot.value_sampler import (
    load_or_create_value_cache,
    sample_database_values,
)


def demo_tables_entry() -> dict:
    return {
        "db_id": "demo",
        "table_names_original": ["singer", "concert"],
        "column_names_original": [
            [-1, "*"],
            [0, "singer_id"],
            [0, "name"],
            [0, "password"],
            [1, "concert_id"],
            [1, "singer_id"],
        ],
        "column_types": [
            "text",
            "number",
            "text",
            "text",
            "number",
            "number",
        ],
        "primary_keys": [1, 4],
        "foreign_keys": [[5, 1]],
    }


class SpiderPreparationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / "demo.sqlite"
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE singer (
                    singer_id INTEGER PRIMARY KEY,
                    name TEXT,
                    password TEXT
                );
                CREATE TABLE concert (
                    concert_id INTEGER PRIMARY KEY,
                    singer_id INTEGER,
                    FOREIGN KEY (singer_id)
                        REFERENCES singer(singer_id)
                );
                INSERT INTO singer VALUES (2, 'Zoë', 'secret-2');
                INSERT INTO singer VALUES (1, 'Alice', 'secret-1');
                INSERT INTO singer VALUES (3, 'A very long singer name', 'x');
                INSERT INTO concert VALUES (10, 1);
                """
            )
            connection.commit()

        self.official = parse_spider_schema_entry(demo_tables_entry())

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_official_schema_matches_sqlite(self) -> None:
        sqlite_schema = parse_sqlite_schema("demo", self.database_path)
        errors = compare_spider_and_sqlite_schema(
            self.official.database,
            sqlite_schema,
        )
        self.assertEqual(errors, [])

    def test_value_sampling_is_deterministic_and_hides_sensitive_columns(
        self,
    ) -> None:
        values = sample_database_values(
            self.official.database,
            self.database_path,
            max_values_per_column=2,
            max_text_length=8,
        )
        self.assertEqual(values["singer"]["singer_id"], [1, 2])
        self.assertEqual(
            values["singer"]["name"],
            ["A very l", "Alice"],
        )
        self.assertNotIn("password", values["singer"])

        cache_root = self.root / "cache"
        first = load_or_create_value_cache(
            self.official.database,
            self.database_path,
            cache_root,
            max_values_per_column=2,
            max_text_length=8,
        )
        second = load_or_create_value_cache(
            self.official.database,
            self.database_path,
            cache_root,
            max_values_per_column=2,
            max_text_length=8,
        )
        self.assertEqual(first, second)
        payload = json.loads(
            (cache_root / "demo.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["version"], 1)

    def test_third_schema_style_contains_example_values(self) -> None:
        output = serialize_schema(
            self.official.database,
            style="structured_with_values",
            example_values={"singer": {"name": ["Alice", "Zoë"]}},
        )
        self.assertIn('examples=["Alice", "Zoë"]', output)

        limited = serialize_schema(
            self.official.database,
            style="structured_with_values",
            example_values={
                "singer": {
                    "name": ["Alice", "Zoë"],
                    "singer_id": [1, 2],
                }
            },
            max_example_characters=12,
        )
        self.assertLessEqual(limited.count("examples="), 1)

        with self.assertRaises(ValueError):
            serialize_schema(
                self.official.database,
                style="structured_with_values",
            )

    def test_schema_link_handles_nested_sql(self) -> None:
        nested_sql = {
            "select": [False, [[0, [0, [0, 2, False], None]]]],
            "from": {"table_units": [["table_unit", 0]], "conds": []},
            "where": [
                [
                    False,
                    8,
                    [0, [0, 1, False], None],
                    {
                        "select": [
                            False,
                            [[0, [0, [0, 5, False], None]]],
                        ],
                        "from": {
                            "table_units": [["table_unit", 1]],
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
                    None,
                ]
            ],
            "groupBy": [],
            "having": [],
            "orderBy": [],
            "limit": None,
            "intersect": None,
            "union": None,
            "except": None,
        }
        link = extract_schema_link(nested_sql, self.official)

        self.assertEqual(link.tables, ("singer", "concert"))
        self.assertEqual(
            link.columns,
            (
                "singer.singer_id",
                "singer.name",
                "concert.singer_id",
            ),
        )
        rendered = serialize_schema_link(link)
        self.assertIn("tables: singer, concert", rendered)
        self.assertIn("columns: singer.singer_id", rendered)


if __name__ == "__main__":
    unittest.main()
