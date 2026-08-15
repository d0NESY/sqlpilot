"""Schema 解析和序列化的单元测试。"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from sqlpilot.schema_parser import parse_sqlite_schema
from sqlpilot.schema_serializer import serialize_schema


class SchemaTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name) / "demo.sqlite"
        )

        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE stadium (
                    stadium_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                );

                CREATE TABLE concert (
                    concert_id INTEGER PRIMARY KEY,
                    stadium_id INTEGER NOT NULL,
                    year INTEGER,
                    FOREIGN KEY (stadium_id)
                        REFERENCES stadium(stadium_id)
                );
                """
            )
            connection.commit()

        self.schema = parse_sqlite_schema(
            db_id="demo",
            db_path=self.database_path,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_parser_reads_tables_primary_keys_and_foreign_keys(self) -> None:
        self.assertEqual(
            [table.name for table in self.schema.tables],
            ["concert", "stadium"],
        )

        concert = self.schema.tables[0]
        concert_id = concert.columns[0]
        self.assertEqual(concert_id.name, "concert_id")
        self.assertTrue(concert_id.is_primary_key)

        self.assertEqual(len(self.schema.foreign_keys), 1)
        foreign_key = self.schema.foreign_keys[0]
        self.assertEqual(foreign_key.source_table, "concert")
        self.assertEqual(foreign_key.source_column, "stadium_id")
        self.assertEqual(foreign_key.target_table, "stadium")
        self.assertEqual(foreign_key.target_column, "stadium_id")

    def test_structured_serializer(self) -> None:
        output = serialize_schema(self.schema, style="structured")

        self.assertIn("Database: demo", output)
        self.assertIn("Table: concert", output)
        self.assertIn("concert_id: INTEGER, PRIMARY KEY", output)
        self.assertIn(
            "concert.stadium_id -> stadium.stadium_id",
            output,
        )

    def test_ddl_serializer(self) -> None:
        output = serialize_schema(self.schema, style="ddl")

        self.assertIn("CREATE TABLE concert", output)
        self.assertIn("concert_id INTEGER PRIMARY KEY", output)
        self.assertIn(
            "FOREIGN KEY (stadium_id) "
            "REFERENCES stadium (stadium_id)",
            output,
        )

    def test_unknown_style_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            serialize_schema(self.schema, style="unknown")


if __name__ == "__main__":
    unittest.main()
