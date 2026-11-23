"""Batch Video Ingestion CLI for AI CCTV Investigator.

Scans raw `.mp4`/`.avi` footage under `data/raw_videos/`, executes:
1. Stage 1 (`PersonDetector` + `PedestrianTracker` + `TrackletCropSampler`) over 3.0s windows.
2. Stage 2/3 (`EmbedExtractor` from `apps/dl-core`, or deterministic `--mock-encoder`).
3. Stage 4 (`FAISSIndexManager` / `IndexHNSWFlat`) persistence to `data/faiss_store/cctv_hnsw.index`,
   `data/tracklets/tracklet_manifest.json`, and `data/tracklets/embeddings_512d.npy`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Ensure local vision-pipeline, shared-contracts, and dl-core modules are importable
VISION_SRC = Path(__file__).resolve().parent
REPO_ROOT = VISION_SRC.parents[2]
SHARED_CONTRACTS_PY = REPO_ROOT / "packages" / "shared-contracts" / "python"
DL_CORE_SRC = REPO_ROOT / "apps" / "dl-core" / "src"

for _p in (str(VISION_SRC), str(SHARED_CONTRACTS_PY), str(DL_CORE_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from constants import (  # noqa: E402
    EMBED_DIM,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    TRACKLET_WIN_SEC,
)
from crop_sampler import TrackletCropSampler  # noqa: E402
from detector import PersonDetector, SyntheticBoxDetector  # noqa: E402
from schemas import TrackletMetadata  # noqa: E402
from tracker import PedestrianTracker  # noqa: E402

logger = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".mp4", ".avi"}


def resolve_repo_path(raw_path: Union[str, Path]) -> Path:
    """Resolve relative paths against repository root."""
    p = Path(raw_path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_cam_map(cam_map_path: Optional[Union[str, Path]]) -> Dict[str, str]:
    """Load optional JSON filename-to-camera-ID mapping."""
    if cam_map_path is None:
        return {}
    resolved = resolve_repo_path(cam_map_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Camera mapping JSON not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("cam-map JSON must be an object mapping filename -> cam_id.")
    return {str(k): str(v) for k, v in data.items()}


def infer_camera_id(video_path: Path, cam_map: Dict[str, str]) -> str:
    """Resolve camera ID from `cam_map` or derive a clean label from video filename."""
    if video_path.name in cam_map:
        return cam_map[video_path.name]
    if video_path.stem in cam_map:
        return cam_map[video_path.stem]
    match = re.search(r"cam[_-]?(\d+)", video_path.stem, flags=re.IGNORECASE)
    if match:
        return f"Cam {int(match.group(1)):02d}"
    return video_path.stem


class MockCropEncoder:
    """Deterministic unit L2-normalized 512-D crop encoder for offline smoke tests."""

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        self.embed_dim = int(embed_dim)

    def encode_crops(self, crop_paths: Sequence[Union[str, Path]]) -> np.ndarray:
        """Hash up to 3 crop images/paths into a `(1, 512)` float32 unit vector."""
        if not crop_paths:
            raise ValueError("At least 1 crop path is required.")
        hasher = hashlib.sha256()
        for item in crop_paths[:3]:
            p = Path(item)
            if p.exists() and p.is_file():
                hasher.update(p.read_bytes())
            else:
                hasher.update(str(item).encode("utf-8"))
        seed = int.from_bytes(hasher.digest()[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal((1, self.embed_dim)).astype(np.float32)
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        return np.ascontiguousarray(vec, dtype=np.float32)


def build_crop_encoder(
    checkpoint_path: Path,
    mock_encoder: bool = False,
    device: Optional[str] = None,
) -> Any:
    """Create `EmbedExtractor` (IRRA fine-tuned or Zero-Shot CLIP fallback) or `MockCropEncoder`."""
    if mock_encoder:
        return MockCropEncoder(embed_dim=EMBED_DIM)

    from embed_extractor import EmbedExtractor  # type: ignore  # noqa: E402

    has_ckpt = checkpoint_path.exists()
    if not has_ckpt:
        logger.warning(
            "Checkpoint '%s' not found; falling back gracefully to Zero-Shot CLIP ViT-B/16.",
            checkpoint_path,
        )
    return EmbedExtractor(
        checkpoint_path=checkpoint_path if has_ckpt else None,
        pretrained="openai",
        device=device,
        use_stripe_head=has_ckpt,
    )


def discover_videos(video_dir: Path) -> List[Path]:
    """Return sorted `.mp4` and `.avi` files in `video_dir`."""
    if not video_dir.exists():
        return []
    return sorted(
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def process_single_video(
    video_path: Path,
    cam_id: str,
    detector: Any,
    sampler: TrackletCropSampler,
    win_sec: float = TRACKLET_WIN_SEC,
) -> List[TrackletMetadata]:
    """Run Stage 1 detection, ByteTrack tracking, and 3.0-second crop sampling on one video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 30.0

    tracker = PedestrianTracker(frame_rate=int(round(fps)))
    buffers: Dict[int, Dict[str, Any]] = {}
    tracklets: List[TrackletMetadata] = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            timestamp = frame_idx / fps
            detections = detector.detect(frame)
            active_tracks = tracker.update(detections, frame)

            for trk in active_tracks:
                tid = int(trk["track_id"])
                buf = buffers.get(tid)
                if buf is None:
                    buf = {"win_start": timestamp, "frames": []}
                    buffers[tid] = buf

                buf["frames"].append(
                    {
                        "frame": frame.copy(),
                        "bbox": trk["bbox"],
                        "timestamp": timestamp,
                    }
                )

                if (timestamp - buf["win_start"]) >= win_sec and buf["frames"]:
                    meta = sampler.process_tracklet(
                        track_id=tid,
                        cam_id=cam_id,
                        video_id=video_path.name,
                        frames_buffer=buf["frames"],
                        start_sec=buf["win_start"],
                        end_sec=buf["win_start"] + win_sec,
                    )
                    tracklets.append(meta)
                    buffers[tid] = {"win_start": timestamp, "frames": []}

            frame_idx += 1
    finally:
        cap.release()

    # Flush trailing partial windows
    for tid, buf in buffers.items():
        if buf["frames"]:
            meta = sampler.process_tracklet(
                track_id=tid,
                cam_id=cam_id,
                video_id=video_path.name,
                frames_buffer=buf["frames"],
                start_sec=buf["win_start"],
                end_sec=max(buf["win_start"] + 0.1, buf["frames"][-1]["timestamp"]),
            )
            tracklets.append(meta)

    return tracklets


def build_and_save_faiss_index(
    embeddings: np.ndarray,
    index_path: Path,
) -> int:
    """Build `IndexHNSWFlat(512, 32, METRIC_INNER_PRODUCT)` and write binary index to disk."""
    import faiss  # type: ignore

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexHNSWFlat(EMBED_DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    if embeddings.shape[0] > 0:
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    faiss.write_index(index, str(index_path))
    return int(index.ntotal)


def _resolve_disk_crop_paths(crop_paths: Sequence[str], crops_dir: Path) -> List[str]:
    resolved: List[str] = []
    for rel in crop_paths:
        candidate_repo = REPO_ROOT / rel
        if candidate_repo.exists():
            resolved.append(str(candidate_repo))
        else:
            candidate_local = crops_dir / Path(rel).name
            resolved.append(str(candidate_local))
    return resolved


def ingest_videos(
    video_dir: Union[str, Path] = "data/raw_videos",
    checkpoint_path: Union[str, Path] = "apps/dl-core/checkpoints/irra_stripe_best.pt",
    cam_map_path: Optional[Union[str, Path]] = None,
    manifest_path: Union[str, Path] = "data/tracklets/tracklet_manifest.json",
    embeddings_path: Union[str, Path] = "data/tracklets/embeddings_512d.npy",
    index_path: Union[str, Path] = "data/faiss_store/cctv_hnsw.index",
    crops_dir: Union[str, Path] = "data/tracklets/crops",
    mock_encoder: bool = False,
    mock_detector: bool = False,
    device: str = "cpu",
    detector: Optional[Any] = None,
    encoder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute full batch ingestion across raw videos and persist Stage 4 FAISS store."""
    v_dir = resolve_repo_path(video_dir)
    ckpt_p = resolve_repo_path(checkpoint_path)
    man_p = resolve_repo_path(manifest_path)
    emb_p = resolve_repo_path(embeddings_path)
    idx_p = resolve_repo_path(index_path)
    crp_dir = resolve_repo_path(crops_dir)

    cam_map = load_cam_map(cam_map_path)
    videos = discover_videos(v_dir)

    if detector is not None:
        det = detector
    elif mock_detector:
        det = SyntheticBoxDetector()
    else:
        det = PersonDetector(device=device)
    sampler = TrackletCropSampler(crops_dir=crp_dir)
    enc = encoder or build_crop_encoder(
        checkpoint_path=ckpt_p, mock_encoder=mock_encoder, device=device
    )

    all_tracklets: List[TrackletMetadata] = []
    for video_path in videos:
        cam_id = infer_camera_id(video_path, cam_map)
        all_tracklets.extend(
            process_single_video(
                video_path=video_path,
                cam_id=cam_id,
                detector=det,
                sampler=sampler,
                win_sec=TRACKLET_WIN_SEC,
            )
        )

    vectors: List[np.ndarray] = []
    for meta in all_tracklets:
        disk_crops = _resolve_disk_crop_paths(meta.crop_paths, crp_dir)
        vec = enc.encode_crops(disk_crops)
        vec_2d = np.asarray(vec, dtype=np.float32).reshape(1, EMBED_DIM)
        norms = np.linalg.norm(vec_2d, axis=1, keepdims=True)
        vec_2d = vec_2d / norms
        assert np.allclose(np.linalg.norm(vec_2d, axis=1), 1.0, atol=1e-5)
        vectors.append(vec_2d)

    if vectors:
        embeddings = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
    else:
        embeddings = np.empty((0, EMBED_DIM), dtype=np.float32)

    man_p.parent.mkdir(parents=True, exist_ok=True)
    with open(man_p, "w", encoding="utf-8") as f:
        json.dump([m.model_dump() for m in all_tracklets], f, indent=2)

    emb_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_p, embeddings)

    indexed_total = build_and_save_faiss_index(embeddings, idx_p)

    return {
        "videos_processed": len(videos),
        "total_tracklets": len(all_tracklets),
        "indexed_vectors": indexed_total,
        "manifest_path": str(man_p),
        "embeddings_path": str(emb_p),
        "index_path": str(idx_p),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for batch ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest raw CCTV videos into Stage 1 tracklets, Stage 2/3 embeddings, and Stage 4 FAISS index."
    )
    parser.add_argument("--video-dir", type=str, default="data/raw_videos")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="apps/dl-core/checkpoints/irra_stripe_best.pt",
    )
    parser.add_argument("--cam-map", type=str, default=None)
    parser.add_argument(
        "--manifest-path", type=str, default="data/tracklets/tracklet_manifest.json"
    )
    parser.add_argument(
        "--embeddings-path", type=str, default="data/tracklets/embeddings_512d.npy"
    )
    parser.add_argument(
        "--index-path", type=str, default="data/faiss_store/cctv_hnsw.index"
    )
    parser.add_argument("--crops-dir", type=str, default="data/tracklets/crops")
    parser.add_argument(
        "--mock-encoder",
        action="store_true",
        help="Use deterministic mock 512-D crop encoder for fast offline tests.",
    )
    parser.add_argument(
        "--mock-detector",
        action="store_true",
        help="Use contour-based synthetic box detector for offline smoke tests.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)
    summary = ingest_videos(
        video_dir=args.video_dir,
        checkpoint_path=args.checkpoint,
        cam_map_path=args.cam_map,
        manifest_path=args.manifest_path,
        embeddings_path=args.embeddings_path,
        index_path=args.index_path,
        crops_dir=args.crops_dir,
        mock_encoder=args.mock_encoder,
        mock_detector=args.mock_detector,
        device=args.device,
    )
    print(
        f"Batch ingestion complete: videos={summary['videos_processed']}, "
        f"tracklets={summary['total_tracklets']}, indexed={summary['indexed_vectors']}"
    )


if __name__ == "__main__":
    main()
