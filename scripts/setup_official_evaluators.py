"""安装固定 commit 的官方评估器，并可安全解压 TSA 数据库。"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "tools" / "official_evaluation"
REPOSITORIES = {
    "spider": {
        "url": "https://github.com/taoyds/spider.git",
        "commit": "b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c",
        "required": ("evaluation.py", "process_sql.py"),
    },
    "test-suite-sql-eval": {
        "url": "https://github.com/taoyds/test-suite-sql-eval.git",
        "commit": "e97acc546ecbee8fa27fa8dbf025ef61493a876c",
        "required": (
            "evaluation.py",
            "process_sql.py",
            "exec_eval.py",
            "exec_subprocess.py",
        ),
    },
}
TEST_SUITE_DOWNLOAD = (
    "https://drive.google.com/file/d/"
    "1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w/view?usp=sharing"
)


def _run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"命令失败：{command}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return result.stdout.strip()


def _install_repository(root: Path, name: str, spec: dict[str, object]) -> dict[str, str]:
    destination = root / name
    commit = str(spec["commit"])
    if destination.exists():
        if not (destination / ".git").is_dir():
            raise FileExistsError(f"目标已存在但不是 Git 仓库：{destination}")
        actual = _run(["git", "rev-parse", "HEAD"], cwd=destination)
        if actual != commit:
            raise RuntimeError(
                f"{name} commit 不一致：expected={commit}, actual={actual}"
            )
    else:
        destination.mkdir(parents=True)
        _run(["git", "init"], cwd=destination)
        _run(["git", "remote", "add", "origin", str(spec["url"])], cwd=destination)
        _run(["git", "fetch", "--depth", "1", "origin", commit], cwd=destination)
        _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=destination)
    for relative in spec["required"]:
        if not (destination / str(relative)).is_file():
            raise FileNotFoundError(f"{name} 缺少官方文件：{relative}")
    return {"path": str(destination), "url": str(spec["url"]), "commit": commit}


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        root = destination.resolve()
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"压缩包包含越界路径：{member.filename}")
        bundle.extractall(destination)

    candidates: list[Path] = []
    for concert_dir in destination.rglob("concert_singer"):
        relative_parts = concert_dir.relative_to(destination).parts
        if "__MACOSX" in relative_parts:
            continue
        sqlite_files = (
            path
            for path in concert_dir.rglob("*.sqlite")
            if "__MACOSX" not in path.relative_to(destination).parts
            and not path.name.startswith("._")
        )
        if concert_dir.is_dir() and any(sqlite_files):
            candidates.append(concert_dir.parent)
    unique = sorted({item.resolve() for item in candidates})
    if len(unique) != 1:
        raise RuntimeError(
            "无法唯一定位 Test Suite database 根目录；"
            f"候选={list(map(str, unique))}"
        )
    return unique[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定 Spider 官方评估器版本")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--test-suite-archive",
        type=Path,
        default=None,
        help="从官方 Google Drive 手动下载的 testsuitedatabases.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    installed = {
        name: _install_repository(root, name, spec)
        for name, spec in REPOSITORIES.items()
    }
    test_suite_database = None
    if args.test_suite_archive is not None:
        archive = args.test_suite_archive.resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"找不到 Test Suite 压缩包：{archive}")
        test_suite_database = _safe_extract(
            archive,
            root / "test_suite_databases",
        )
    result = {
        "official_repositories": installed,
        "test_suite_download": TEST_SUITE_DOWNLOAD,
        "test_suite_database_root": (
            None if test_suite_database is None else str(test_suite_database)
        ),
        "next_step": (
            "下载官方 testsuitedatabases.zip 后用 "
            "--test-suite-archive 参数再次执行"
            if test_suite_database is None
            else "官方评估器与 TSA 数据库已就绪"
        ),
    }
    (root / "setup_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
