"""Hugging Face 固定 commit 离线解析测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlpilot.hub import resolve_model_source


class HubResolutionTestCase(unittest.TestCase):
    def test_offline_mode_requires_and_uses_exact_snapshot(self) -> None:
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            snapshot = (
                cache
                / "models--Qwen--Demo"
                / "snapshots"
                / revision
            )
            snapshot.mkdir(parents=True)
            with patch.dict(
                os.environ,
                {
                    "HF_HUB_OFFLINE": "1",
                    "HF_HUB_CACHE": str(cache),
                },
            ):
                source = resolve_model_source("Qwen/Demo", revision)
                self.assertTrue(source.offline)
                self.assertEqual(source.source, str(snapshot.resolve()))
                self.assertEqual(source.resolved_revision, revision)
                with self.assertRaises(FileNotFoundError):
                    resolve_model_source("Qwen/Missing", revision)


if __name__ == "__main__":
    unittest.main()
