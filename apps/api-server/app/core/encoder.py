"""Query and tracklet crop embedding service for FastAPI inference.

Uses `EmbedExtractor` from `apps/dl-core` when weights are available, and supports
a fast deterministic mock encoder controlled via `CCTV_MOCK_ENCODER=1`.
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
from PIL import Image

from core.config import EMBED_DIM, REPO_ROOT, ensure_dl_core_on_path

logger = logging.getLogger(__name__)


def _deterministic_unit_vector(seed_bytes: bytes, dim: int = EMBED_DIM) -> np.ndarray:
    """Hash arbitrary bytes into a reproducible `(1, dim)` unit L2-norm float32 array."""
    digest = hashlib.sha256(seed_bytes).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal((1, dim)).astype(np.float32)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return np.ascontiguousarray(vec / norm, dtype=np.float32)


class EncoderService:
    """Unified 512-D embedding encoder with IRRA/CLIP + deterministic mock modes."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        mock_mode: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.mock_mode = bool(mock_mode)
        self.checkpoint_path = checkpoint_path
        self.extractor: Optional[Any] = None
        self.model_status: str = "mock"

        if not self.mock_mode:
            self._init_dl_core_extractor(checkpoint_path=checkpoint_path, device=device)

    def _init_dl_core_extractor(
        self, checkpoint_path: Optional[Path], device: Optional[str]
    ) -> None:
        ensure_dl_core_on_path()
        try:
            from embed_extractor import EmbedExtractor  # type: ignore

            has_ckpt = checkpoint_path is not None and checkpoint_path.exists()
            self.extractor = EmbedExtractor(
                checkpoint_path=checkpoint_path if has_ckpt else None,
                pretrained="openai",
                device=device,
                use_stripe_head=has_ckpt,
            )
            self.model_status = "irra-stripe" if has_ckpt else "clip-zeroshot"
        except Exception as exc:
            logger.warning(
                "Falling back to deterministic mock encoder (failed to initialize EmbedExtractor: %s)",
                exc,
            )
            self.mock_mode = True
            self.extractor = None
            self.model_status = "mock-fallback"

    def encode_text_query(self, query: str) -> np.ndarray:
        """Encode a natural language query string into `(1, 512)` float32 unit vector."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Search query string must be non-empty.")

        if self.mock_mode or self.extractor is None:
            return _deterministic_unit_vector(clean_query.lower().encode("utf-8"))
        return self.extractor.encode_text_query(clean_query)

    def encode_crops(
        self,
        crops: Sequence[Union[str, Path, bytes, Image.Image]],
        fallback_seed: Optional[str] = None,
    ) -> np.ndarray:
        """Encode up to 3 crop paths, PIL images, or raw image bytes into `(1, 512)` float32."""
        if not crops:
            if fallback_seed:
                return _deterministic_unit_vector(fallback_seed.encode("utf-8"))
            raise ValueError("At least one crop path or image is required.")

        if self.mock_mode or self.extractor is None:
            hasher = hashlib.sha256()
            if fallback_seed:
                hasher.update(fallback_seed.encode("utf-8"))
            for item in crops[:3]:
                if isinstance(item, bytes):
                    hasher.update(item)
                elif isinstance(item, Image.Image):
                    hasher.update(item.tobytes())
                else:
                    path = Path(item)
                    if not path.is_absolute() and not path.exists():
                        path = REPO_ROOT / path
                    if path.exists() and path.is_file():
                        hasher.update(path.read_bytes())
                    else:
                        hasher.update(str(item).encode("utf-8"))
            return _deterministic_unit_vector(hasher.digest())

        prepared = []
        for item in crops[:3]:
            if isinstance(item, bytes):
                with Image.open(io.BytesIO(item)) as im:
                    prepared.append(im.convert("RGB"))
            else:
                prepared.append(item)
        return self.extractor.encode_crops(prepared)
