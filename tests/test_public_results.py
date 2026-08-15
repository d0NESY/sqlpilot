"""公开结果摘要的内部一致性和脱敏边界测试。"""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


class PublicResultsTestCase(unittest.TestCase):
    def test_metrics_json_and_csv_are_consistent(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results_summary"
        summary = json.loads(
            (root / "official_metrics.json").read_text(encoding="utf-8")
        )
        with (root / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            csv_rows = {row["experiment"]: row for row in csv.DictReader(stream)}

        self.assertEqual(summary["benchmark"]["record_count"], 1034)
        self.assertEqual(len(summary["experiments"]), 5)
        self.assertEqual(set(csv_rows), {item["name"] for item in summary["experiments"]})
        for item in summary["experiments"]:
            row = csv_rows[item["name"]]
            self.assertEqual(int(row["record_count"]), item["record_count"])
            for metric in ("em", "ex", "tsa"):
                self.assertEqual(
                    float(row[f"{metric}_percent"]),
                    item["metrics_percent"][metric],
                )

    def test_public_summary_contains_no_server_paths(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results_summary"
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in root.iterdir()
            if path.is_file()
        ).lower()
        for forbidden in ("/home/", "\\home\\", "zhengshuo", "poweredge"):
            self.assertNotIn(forbidden, combined)


if __name__ == "__main__":
    unittest.main()
