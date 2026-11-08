"""Generate 500 synthetic 512-D L2-normalized tracklet embeddings and metadata.

Produces mock artifacts for parallel API, UI, and ML development.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

# Resolve repo root directory
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
TRACKLETS_DIR = DATA_DIR / "tracklets"
CROPS_DIR = TRACKLETS_DIR / "crops"
FAISS_DIR = DATA_DIR / "faiss_store"

NUM_TRACKLETS = 500
EMBED_DIM = 512
TRACKLET_WIN_SEC = 3.0
TOP_CROPS = 3

CAMERAS = [
    "Cam 01 (Main Gate)",
    "Cam 02 (Admin Corridor)",
    "Cam 03 (Library Walkway)",
    "Cam 04 (Cafeteria)",
]


def main() -> None:
    # 1. Ensure target directories exist
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Verified directories:\n  - {CROPS_DIR}\n  - {FAISS_DIR}")

    # 2. Generate 500 synthetic 512-D vectors using NumPy
    np.random.seed(42)  # Deterministic mock generation
    raw_vectors = np.random.randn(NUM_TRACKLETS, EMBED_DIM).astype(np.float32)

    # Strictly L2-normalize all vectors (np.linalg.norm == 1.0)
    norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
    normalized_vectors = raw_vectors / norms

    # Invariant verification: unit norm
    vector_norms = np.linalg.norm(normalized_vectors, axis=1)
    assert np.allclose(vector_norms, 1.0, atol=1e-5), "Vectors must strictly have unit L2 norm!"
    print(f"Generated {NUM_TRACKLETS} vectors of dimension {EMBED_DIM}. L2-norm verification passed.")

    # 3. Create metadata for all 500 tracklets across the 4 cameras
    manifest = []
    for idx in range(NUM_TRACKLETS):
        cam_idx = idx % len(CAMERAS)
        cam_id = CAMERAS[cam_idx]
        cam_code = f"cam{cam_idx + 1:02d}"
        tracklet_id = f"{cam_code}_t{idx:04d}"
        video_id = f"{cam_code}_feed_sample.mp4"
        person_track_id = (idx % 50) + 1  # Simulated recurring tracks
        start_sec = round((idx % 120) * 1.5, 2)
        end_sec = round(start_sec + TRACKLET_WIN_SEC, 2)

        crop_paths = [
            f"data/tracklets/crops/{tracklet_id}_c{c + 1}.jpg"
            for c in range(TOP_CROPS)
        ]

        metadata = {
            "tracklet_id": tracklet_id,
            "video_id": video_id,
            "cam_id": cam_id,
            "person_track_id": person_track_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "crop_paths": crop_paths,
        }
        manifest.append(metadata)

    # 4. Save metadata list to data/tracklets/tracklet_manifest.json (indent=2)
    manifest_path = TRACKLETS_DIR / "tracklet_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved tracklet manifest ({len(manifest)} items) to: {manifest_path}")

    # 5. Save normalized vectors to data/tracklets/embeddings_512d.npy
    embeddings_path = TRACKLETS_DIR / "embeddings_512d.npy"
    np.save(embeddings_path, normalized_vectors)
    print(f"Saved normalized embeddings shape {normalized_vectors.shape} to: {embeddings_path}")

    # 6. FAISS index creation (if faiss installed)
    index_path = FAISS_DIR / "cctv_hnsw.index"
    try:
        import faiss  # type: ignore

        print(f"Found FAISS version: {faiss.__version__}. Building IndexHNSWFlat index...")
        index = faiss.IndexHNSWFlat(EMBED_DIM, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 128
        index.hnsw.efSearch = 64
        index.add(normalized_vectors)
        faiss.write_index(index, str(index_path))
        print(f"Successfully built and wrote FAISS index with {index.ntotal} vectors to: {index_path}")
    except ImportError as e:
        print(
            f"[NOTICE] faiss is not installed in the current Python environment ({e}).\n"
            "Skipping FAISS binary index generation. Run 'pip install faiss-cpu' to enable."
        )


if __name__ == "__main__":
    main()
