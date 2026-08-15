"""在线核对 Hugging Face commit，或严格使用固定 commit 的本地缓存。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelSource:
    source: str
    resolved_revision: str
    model_license: str | None
    offline: bool


def hub_offline_requested() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _TRUE_VALUES


def _hub_cache_root() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return (Path(hf_home).expanduser() / "hub").resolve()
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def _cached_snapshot(model_name: str, revision: str) -> Path:
    repository = "models--" + model_name.replace("/", "--")
    return _hub_cache_root() / repository / "snapshots" / revision


def resolve_model_source(model_name: str, revision: str) -> ModelSource:
    """联网时核对 commit；离线时要求同一 commit 的 snapshot 已存在。"""

    if hub_offline_requested():
        snapshot = _cached_snapshot(model_name, revision)
        if not snapshot.is_dir():
            raise FileNotFoundError(
                "HF_HUB_OFFLINE=1，但找不到固定 commit 的模型缓存："
                f"{snapshot}"
            )
        return ModelSource(
            source=str(snapshot),
            resolved_revision=revision,
            model_license=None,
            offline=True,
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("缺少 huggingface-hub 依赖") from error

    model_info = HfApi().model_info(model_name, revision=revision)
    resolved_revision = model_info.sha
    if not resolved_revision.startswith(revision):
        raise ValueError(
            "Hub 返回的模型 commit 与配置 revision 不匹配："
            f"{resolved_revision}"
        )
    return ModelSource(
        source=model_name,
        resolved_revision=resolved_revision,
        model_license=getattr(
            getattr(model_info, "card_data", None),
            "license",
            None,
        ),
        offline=False,
    )
