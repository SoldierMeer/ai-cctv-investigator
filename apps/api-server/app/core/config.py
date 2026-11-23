"""Repository-anchored settings and shared contracts loader for Stage 4 & 5 API server."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
SHARED_CONTRACTS_PY = REPO_ROOT / "packages" / "shared-contracts" / "python"
DL_CORE_SRC = REPO_ROOT / "apps" / "dl-core" / "src"


def ensure_contracts_on_path() -> Path:
    """Ensure `packages/shared-contracts/python` is importable on `sys.path`."""
    contracts_str = str(SHARED_CONTRACTS_PY)
    if contracts_str not in sys.path:
        sys.path.insert(0, contracts_str)
    return SHARED_CONTRACTS_PY


def ensure_dl_core_on_path() -> Path:
    """Ensure `apps/dl-core/src` is importable on `sys.path`."""
    dl_core_str = str(DL_CORE_SRC)
    if dl_core_str not in sys.path:
        sys.path.insert(0, dl_core_str)
    return DL_CORE_SRC


ensure_contracts_on_path()

from constants import (  # noqa: E402
    EMBED_DIM,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    TARGET_LATENCY_MS,
)


def resolve_repo_path(relative_or_env: Optional[str], default_relative: str) -> Path:
    """Resolve a path relative to repository root unless an absolute path is explicitly passed."""
    raw = relative_or_env if relative_or_env else default_relative
    path = Path(raw)
    if not path.is_absolute():
        return (REPO_ROOT / path).resolve()
    return path


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime configuration anchored to the repository root `data/` directory."""

    repo_root: Path = REPO_ROOT
    faiss_index_path: Path = field(
        default_factory=lambda: resolve_repo_path(
            os.getenv("CCTV_FAISS_INDEX_PATH"), "data/faiss_store/cctv_hnsw.index"
        )
    )
    tracklet_manifest_path: Path = field(
        default_factory=lambda: resolve_repo_path(
            os.getenv("CCTV_TRACKLET_MANIFEST_PATH"), "data/tracklets/tracklet_manifest.json"
        )
    )
    raw_videos_dir: Path = field(
        default_factory=lambda: resolve_repo_path(
            os.getenv("CCTV_RAW_VIDEOS_DIR"), "data/raw_videos"
        )
    )
    crops_dir: Path = field(
        default_factory=lambda: resolve_repo_path(
            os.getenv("CCTV_CROPS_DIR"), "data/tracklets/crops"
        )
    )
    irra_checkpoint_path: Path = field(
        default_factory=lambda: resolve_repo_path(
            os.getenv("CCTV_IRRA_CHECKPOINT_PATH"),
            "apps/dl-core/checkpoints/irra_stripe_best.pt",
        )
    )
    mock_encoder: bool = field(default_factory=lambda: _env_flag("CCTV_MOCK_ENCODER", False))
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])
    embed_dim: int = EMBED_DIM
    hnsw_m: int = HNSW_M
    hnsw_ef_construction: int = HNSW_EF_CONSTRUCTION
    hnsw_ef_search: int = HNSW_EF_SEARCH
    target_latency_ms: float = TARGET_LATENCY_MS


def get_settings() -> Settings:
    """Build fresh `Settings` reading current environment overrides."""
    return Settings()
