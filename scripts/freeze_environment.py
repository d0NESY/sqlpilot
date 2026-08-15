"""在 GPU 冒烟训练通过后生成服务器专属精确锁文件。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把当前 Python 环境精确冻结为 requirements.lock"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "requirements.lock",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    output = args.output.resolve()
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(completed.stdout, encoding="utf-8", newline="\n")
    temporary.replace(output)
    print(f"已写入：{output}")


if __name__ == "__main__":
    main()
