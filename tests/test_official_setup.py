"""官方评估器离线安装与 TSA 安全解压测试。"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "setup_official_evaluators.py"
SPEC = importlib.util.spec_from_file_location("setup_official_evaluators", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


class OfficialSetupTestCase(unittest.TestCase):
    def test_safe_extract_ignores_macos_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "testsuitedatabases.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(
                    "database/concert_singer/concert_singer.sqlite",
                    b"sqlite",
                )
                bundle.writestr(
                    "__MACOSX/database/concert_singer/._concert_singer.sqlite",
                    b"metadata",
                )

            actual = SETUP._safe_extract(archive, root / "extracted")

            self.assertEqual(actual, (root / "extracted" / "database").resolve())

    def test_safe_extract_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape.sqlite", b"unsafe")

            with self.assertRaises(ValueError):
                SETUP._safe_extract(archive, root / "extracted")


if __name__ == "__main__":
    unittest.main()
