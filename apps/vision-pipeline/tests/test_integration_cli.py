"""Integration & CLI tests for batch_ingest.py and live_webcam_daemon.py."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import faiss
from fastapi.testclient import TestClient
import numpy as np
import pytest

VISION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = VISION_ROOT / "src"
SHARED_CONTRACTS = REPO_ROOT / "packages" / "shared-contracts" / "python"
API_APP_DIR = REPO_ROOT / "apps" / "api-server" / "app"

for p in (str(SRC_DIR), str(SHARED_CONTRACTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import batch_ingest  # noqa: E402
from detector import SyntheticBoxDetector  # noqa: E402
import live_webcam_daemon  # noqa: E402
from schemas import TrackletMetadata  # noqa: E402


def _write_synthetic_mp4(video_path: Path, num_frames: int = 35, fps: float = 10.0) -> None:
    """Create a synthetic `.mp4` containing a moving textured pedestrian box via cv2.VideoWriter."""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"VideoWriter failed to open {video_path}"

    try:
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x1 = 30 + i * 4
            y1 = 40
            x2 = x1 + 60
            y2 = 200
            cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), -1)
            cv2.rectangle(frame, (x1 + 10, y1 + 20), (x2 - 10, y1 + 90), (40, 80, 200), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_batch_ingest_cli_synthetic_mp4(tmp_path: Path) -> None:
    """Verify batch_ingest.py end-to-end on a synthetic MP4 with --mock-encoder."""
    video_dir = tmp_path / "raw_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = video_dir / "cam01_sample.mp4"
    _write_synthetic_mp4(mp4_path, num_frames=35, fps=10.0)

    cam_map_file = tmp_path / "cam_map.json"
    cam_map_file.write_text(
        json.dumps({"cam01_sample.mp4": "Cam 01 (Test Gate)"}),
        encoding="utf-8",
    )

    manifest_out = tmp_path / "tracklet_manifest.json"
    embeds_out = tmp_path / "embeddings_512d.npy"
    index_out = tmp_path / "cctv_hnsw.index"
    crops_out = tmp_path / "crops"

    # Unit checks for helpers
    loaded_map = batch_ingest.load_cam_map(cam_map_file)
    assert batch_ingest.infer_camera_id(mp4_path, loaded_map) == "Cam 01 (Test Gate)"
    assert batch_ingest.infer_camera_id(Path("cam03_walk.mp4"), {}) == "Cam 03"
    assert batch_ingest.discover_videos(video_dir) == [mp4_path]

    # Run batch_ingest CLI entrypoint with --mock-encoder and --mock-detector
    batch_ingest.main(
        [
            "--video-dir",
            str(video_dir),
            "--cam-map",
            str(cam_map_file),
            "--manifest-path",
            str(manifest_out),
            "--embeddings-path",
            str(embeds_out),
            "--index-path",
            str(index_out),
            "--crops-dir",
            str(crops_out),
            "--mock-encoder",
            "--mock-detector",
        ]
    )

    assert manifest_out.exists()
    assert embeds_out.exists()
    assert index_out.exists()

    raw_manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert len(raw_manifest) >= 2  # 3.5s video produces 1 full 3.0s window + 1 trailing window
    validated = [TrackletMetadata.model_validate(item) for item in raw_manifest]
    assert all(m.cam_id == "Cam 01 (Test Gate)" for m in validated)
    assert all(m.video_id == "cam01_sample.mp4" for m in validated)

    embeddings = np.load(embeds_out)
    assert embeddings.shape == (len(validated), 512)
    assert embeddings.dtype == np.float32
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)

    faiss_index = faiss.read_index(str(index_out))
    assert faiss_index.ntotal == len(validated)

    # Verify crops were saved at 384x128
    saved_crops = list(crops_out.glob("*.jpg"))
    assert len(saved_crops) >= 3
    sample_img = cv2.imread(str(saved_crops[0]))
    assert sample_img is not None and sample_img.shape == (384, 128, 3)


def test_live_webcam_daemon_dry_run_and_api_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify live_webcam_daemon.py with --dry-run-frames 5 via CLI and live FastAPI TestClient."""
    # 1. Subprocess CLI test with --dry-run-frames 5
    cli_crops = tmp_path / "cli_crops"
    proc = subprocess.run(
        [
            sys.executable,
            str(SRC_DIR / "live_webcam_daemon.py"),
            "--dry-run-frames",
            "5",
            "--crops-dir",
            str(cli_crops),
            "--mock-detector",
            "--skip-upload",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "frames=5" in proc.stdout
    assert len(list(cli_crops.glob("*.jpg"))) >= 3

    # 2. Direct SyntheticBoxDetector + generate_synthetic_frame unit check
    synth_det = SyntheticBoxDetector()
    synth_frame = live_webcam_daemon.generate_synthetic_frame(0)
    boxes = synth_det.detect(synth_frame)
    assert boxes.ndim == 2 and boxes.shape[0] >= 1 and boxes.shape[1] == 6

    # 3. End-to-end live multipart ingest into FastAPI TestClient
    if str(API_APP_DIR) not in sys.path:
        sys.path.insert(0, str(API_APP_DIR))
    from main import create_app  # type: ignore

    monkeypatch.setenv("CCTV_MOCK_ENCODER", "1")
    api_crops = tmp_path / "api_crops"

    with TestClient(create_app()) as client:
        before = client.get("/health").json()["total_indexed_tracklets"]
        summary = live_webcam_daemon.run_live_daemon(
            source="0",
            cam_id="Live Defense Cam",
            dry_run_frames=5,
            crops_dir=api_crops,
            mock_detector=True,
            skip_upload=False,
            http_client=client,
        )
        assert summary["frames_processed"] == 5
        assert summary["tracklets_emitted"] >= 1
        assert summary["uploaded_count"] == summary["tracklets_emitted"]
        assert all(t.cam_id == "Live Defense Cam" for t in summary["tracklets"])

        after = client.get("/health").json()["total_indexed_tracklets"]
        assert after == before + summary["uploaded_count"]
