"""Shared contracts and repository-anchored paths for Stage 2 & 3 (dl-core)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_CONTRACTS_PY = REPO_ROOT / "packages" / "shared-contracts" / "python"
DATA_DIR = REPO_ROOT / "data"
DL_CORE_ROOT = REPO_ROOT / "apps" / "dl-core"
CHECKPOINTS_DIR = DL_CORE_ROOT / "checkpoints"


def ensure_shared_contracts_on_path() -> Path:
    """Ensure `packages/shared-contracts/python` is on `sys.path`."""
    contracts_str = str(SHARED_CONTRACTS_PY)
    if contracts_str not in sys.path:
        sys.path.insert(0, contracts_str)
    return SHARED_CONTRACTS_PY


ensure_shared_contracts_on_path()

from constants import (  # noqa: E402
    EMBED_DIM,
    NUM_HORIZONTAL_STRIPES,
    TOP_CROPS_PER_TRACKLET,
    TRACKLET_WIN_SEC,
)

__all__ = [
    "REPO_ROOT",
    "SHARED_CONTRACTS_PY",
    "DATA_DIR",
    "DL_CORE_ROOT",
    "CHECKPOINTS_DIR",
    "EMBED_DIM",
    "NUM_HORIZONTAL_STRIPES",
    "TOP_CROPS_PER_TRACKLET",
    "TRACKLET_WIN_SEC",
    "ensure_shared_contracts_on_path",
]
