"""FAISS IndexHNSWFlat manager for 512-D unit-normalized tracklet embeddings."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import List, Optional, Union

import faiss
import numpy as np

from core.config import (
    EMBED_DIM,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    ensure_contracts_on_path,
)

ensure_contracts_on_path()
from schemas import SearchResultItem, TrackletMetadata  # noqa: E402


def normalize_embedding(vector: np.ndarray, expected_dim: int = EMBED_DIM) -> np.ndarray:
    """Validate shape/finiteness and L2-normalize vectors to `(N, D)` float32."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D embedding array, got shape {arr.shape}")

    if arr.shape[1] != expected_dim:
        raise ValueError(f"Expected embedding dim {expected_dim}, got {arr.shape[1]}")
    if not np.isfinite(arr).all():
        raise ValueError("Embedding vector contains NaN or Inf values.")

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Cannot L2-normalize zero-norm embedding vector.")

    normalized = arr / norms
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _matches_cam_filter(cam_id: str, cam_filter: str) -> bool:
    """Match exact camera ID or short prefix (e.g., 'Cam 02' matches 'Cam 02 (Admin Corridor)')."""
    cam_norm = cam_id.strip().lower()
    flt_norm = cam_filter.strip().lower()
    if not flt_norm:
        return True
    if cam_norm == flt_norm:
        return True
    return cam_norm.startswith(f"{flt_norm} ") or cam_norm.startswith(f"{flt_norm}(")


class FAISSIndexManager:
    """Thread-safe wrapper over `faiss.IndexHNSWFlat` + `TrackletMetadata` manifest."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hnsw_m: int = HNSW_M,
        ef_construction: int = HNSW_EF_CONSTRUCTION,
        ef_search: int = HNSW_EF_SEARCH,
    ) -> None:
        self.embed_dim = int(embed_dim)
        self.hnsw_m = int(hnsw_m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self._lock = threading.RLock()

        self.index = self._create_empty_index()
        self.metadata: List[TrackletMetadata] = []

    def _create_empty_index(self) -> faiss.IndexHNSWFlat:
        index = faiss.IndexHNSWFlat(self.embed_dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        return index

    @property
    def total_indexed_tracklets(self) -> int:
        """Total number of indexed tracklet vectors."""
        with self._lock:
            return len(self.metadata)

    @property
    def cameras(self) -> List[str]:
        """Sorted list of unique camera identifiers currently indexed."""
        with self._lock:
            return sorted({item.cam_id for item in self.metadata})

    def load(
        self,
        index_path: Union[str, Path],
        manifest_path: Union[str, Path],
    ) -> int:
        """Load persisted FAISS HNSW index and JSON tracklet manifest from disk."""
        idx_p = Path(index_path)
        man_p = Path(manifest_path)

        if not idx_p.exists():
            raise FileNotFoundError(f"FAISS index file not found: {idx_p}")
        if not man_p.exists():
            raise FileNotFoundError(f"Tracklet manifest file not found: {man_p}")

        loaded_index = faiss.read_index(str(idx_p))
        if loaded_index.d != self.embed_dim:
            raise ValueError(
                f"Loaded index dimension ({loaded_index.d}) != expected {self.embed_dim}"
            )
        if hasattr(loaded_index, "hnsw"):
            loaded_index.hnsw.efSearch = self.ef_search

        with open(man_p, "r", encoding="utf-8") as f:
            raw_items = json.load(f)
        loaded_meta = [TrackletMetadata.model_validate(item) for item in raw_items]

        if loaded_index.ntotal != len(loaded_meta):
            raise ValueError(
                f"FAISS index count ({loaded_index.ntotal}) != manifest count ({len(loaded_meta)})"
            )

        with self._lock:
            self.index = loaded_index
            self.metadata = loaded_meta
            return len(self.metadata)

    def save(
        self,
        index_path: Union[str, Path],
        manifest_path: Union[str, Path],
    ) -> None:
        """Persist current FAISS index and metadata manifest to disk."""
        idx_p = Path(index_path)
        man_p = Path(manifest_path)
        idx_p.parent.mkdir(parents=True, exist_ok=True)
        man_p.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            faiss.write_index(self.index, str(idx_p))
            with open(man_p, "w", encoding="utf-8") as f:
                json.dump([item.model_dump() for item in self.metadata], f, indent=2)

    def add(self, vector: np.ndarray, metadata: TrackletMetadata) -> int:
        """Enforce unit L2 norm, append vector + metadata in memory, and return assigned ID."""
        norm_vec = normalize_embedding(vector, expected_dim=self.embed_dim)
        if norm_vec.shape[0] != 1:
            raise ValueError(f"Expected single (1, {self.embed_dim}) vector, got {norm_vec.shape}")
        meta_obj = TrackletMetadata.model_validate(metadata)

        with self._lock:
            assigned_id = len(self.metadata)
            self.index.add(norm_vec)
            self.metadata.append(meta_obj)
            return assigned_id

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        cam_filter: Optional[str] = None,
    ) -> List[SearchResultItem]:
        """Search top-K closest tracklets by inner product (cosine similarity)."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        q_norm = normalize_embedding(query_vector, expected_dim=self.embed_dim)
        if q_norm.shape[0] != 1:
            raise ValueError(f"Expected single query vector, got shape {q_norm.shape}")

        with self._lock:
            ntotal = len(self.metadata)
            if ntotal == 0:
                return []

            has_filter = bool(cam_filter and cam_filter.strip())
            if not has_filter:
                candidate_k = min(top_k, ntotal)
            else:
                # Over-fetch candidates to satisfy `top_k` after camera filtering
                candidate_k = min(ntotal, max(top_k * 8, 250))

            while True:
                scores, indices = self.index.search(q_norm, candidate_k)
                matched: List[tuple[float, TrackletMetadata]] = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= ntotal:
                        continue
                    item = self.metadata[int(idx)]
                    if has_filter and not _matches_cam_filter(item.cam_id, cam_filter or ""):
                        continue
                    matched.append((float(score), item))

                if not has_filter or len(matched) >= top_k or candidate_k >= ntotal:
                    break
                candidate_k = min(ntotal, candidate_k * 2)

            matched.sort(key=lambda pair: pair[0], reverse=True)
            top_matched = matched[:top_k]

            return [
                SearchResultItem(
                    rank=rank_idx,
                    similarity_score=round(score, 6),
                    tracklet=tracklet,
                )
                for rank_idx, (score, tracklet) in enumerate(top_matched, start=1)
            ]
