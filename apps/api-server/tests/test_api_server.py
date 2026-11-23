"""Test suite for Stage 4 & 5A (FAISS HNSW Manager + FastAPI Backend).

Tests:
1. `GET /health` returns 200 and reports `total_indexed_tracklets == 500` from seeded index.
2. `POST /api/v1/search` returns top-K descending results, respects `cam_filter`, and runs < 400 ms.
3. `GET /api/v1/video/{video_id}` returns HTTP 206 Partial Content with valid Content-Range,
   404 on missing files, and blocks path traversal (`..`).
4. `POST /api/v1/ingest/live` (JSON and multipart) dynamically updates FAISS and increments total count.
5. `FAISSIndexManager` unit norm enforcement, save/load persistence, and config helpers.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

API_SERVER_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = API_SERVER_ROOT / "app"
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from core.config import (  # noqa: E402
    EMBED_DIM,
    TARGET_LATENCY_MS,
    ensure_contracts_on_path,
    ensure_dl_core_on_path,
    get_settings,
    resolve_repo_path,
)
from core.faiss_index import FAISSIndexManager, normalize_embedding  # noqa: E402
from main import create_app  # noqa: E402
from routers.video_router import parse_byte_range, resolve_safe_video_path  # noqa: E402

ensure_contracts_on_path()
from schemas import TrackletMetadata  # noqa: E402


@pytest.fixture()
def seeded_client(monkeypatch: pytest.MonkeyPatch):
    """FastAPI TestClient backed by the seeded 500-vector FAISS index and mock encoder."""
    monkeypatch.setenv("CCTV_MOCK_ENCODER", "1")
    application = create_app()
    with TestClient(application) as client:
        yield client


# ============================================================================
# Test 1: GET /health returns 200 and reports seeded 500-tracklet index
# ============================================================================
def test_health_endpoint_seeded_index(seeded_client: TestClient) -> None:
    response = seeded_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["total_indexed_tracklets"] == 500
    assert payload["model_status"] == "mock"
    assert len(payload["cameras"]) == 4
    assert "Cam 01 (Main Gate)" in payload["cameras"]


# ============================================================================
# Test 2: POST /api/v1/search top-K ordering, cam_filter, and < 400 ms latency
# ============================================================================
def test_search_endpoint_ranking_filter_and_latency(seeded_client: TestClient) -> None:
    # Unfiltered top-10 search
    resp = seeded_client.post(
        "/api/v1/search",
        json={"query": "Person wearing a red jacket and dark jeans", "top_k": 10},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_indexed_tracklets"] == 500
    assert data["latency_ms"] < TARGET_LATENCY_MS
    results = data["results"]
    assert len(results) == 10

    scores = [item["similarity_score"] for item in results]
    ranks = [item["rank"] for item in results]
    assert ranks == list(range(1, 11))
    assert scores == sorted(scores, reverse=True)

    # Filtered search with exact camera name
    resp_exact = seeded_client.post(
        "/api/v1/search",
        json={
            "query": "Pedestrian carrying a black backpack",
            "top_k": 5,
            "cam_filter": "Cam 02 (Admin Corridor)",
        },
    )
    assert resp_exact.status_code == 200
    exact_data = resp_exact.json()
    assert exact_data["latency_ms"] < TARGET_LATENCY_MS
    assert len(exact_data["results"]) == 5
    assert all(
        item["tracklet"]["cam_id"] == "Cam 02 (Admin Corridor)"
        for item in exact_data["results"]
    )

    # Filtered search with short camera prefix ("Cam 03")
    resp_prefix = seeded_client.post(
        "/api/v1/search",
        json={
            "query": "Student walking past library entrance",
            "top_k": 5,
            "cam_filter": "Cam 03",
        },
    )
    assert resp_prefix.status_code == 200
    prefix_data = resp_prefix.json()
    assert len(prefix_data["results"]) == 5
    assert all(
        item["tracklet"]["cam_id"].startswith("Cam 03")
        for item in prefix_data["results"]
    )


# ============================================================================
# Test 3: GET /api/v1/video/{video_id} HTTP 206 Partial Content + security
# ============================================================================
def test_video_byte_range_streaming_and_security(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dummy_bytes = bytes((i % 256 for i in range(4096)))
    dummy_video = tmp_path / "dummy_cam01.mp4"
    dummy_video.write_bytes(dummy_bytes)

    monkeypatch.setenv("CCTV_MOCK_ENCODER", "1")
    monkeypatch.setenv("CCTV_RAW_VIDEOS_DIR", str(tmp_path))

    with TestClient(create_app()) as client:
        # 1. Partial Content byte-range request (bytes=0-1023)
        resp_206 = client.get(
            "/api/v1/video/dummy_cam01.mp4",
            headers={"Range": "bytes=0-1023"},
        )
        assert resp_206.status_code == 206
        assert resp_206.headers.get("accept-ranges") == "bytes"
        assert resp_206.headers.get("content-range") == "bytes 0-1023/4096"
        assert resp_206.headers.get("content-length") == "1024"
        assert resp_206.content == dummy_bytes[:1024]

        # 2. Full file request without Range header
        resp_200 = client.get("/api/v1/video/dummy_cam01.mp4")
        assert resp_200.status_code == 200
        assert resp_200.content == dummy_bytes

        # 3. Missing video returns 404
        resp_404 = client.get("/api/v1/video/nonexistent_cam.mp4")
        assert resp_404.status_code == 404

        # 4. Path traversal blocked
        resp_trav = client.get("/api/v1/video/..%5Csecret.mp4")
        assert resp_trav.status_code == 400

        # 5. Unsatisfiable range returns 416
        resp_416 = client.get(
            "/api/v1/video/dummy_cam01.mp4",
            headers={"Range": "bytes=9999-10000"},
        )
        assert resp_416.status_code == 416

    # Unit checks for parse_byte_range and resolve_safe_video_path
    assert parse_byte_range("bytes=-512", 4096) == (3584, 4095)
    assert resolve_safe_video_path(tmp_path, "dummy_cam01.mp4") == dummy_video.resolve()


# ============================================================================
# Test 4: POST /api/v1/ingest/live dynamically increments FAISS index
# ============================================================================
def test_live_ingest_json_and_multipart(seeded_client: TestClient) -> None:
    initial_health = seeded_client.get("/health").json()
    initial_count = initial_health["total_indexed_tracklets"]
    assert initial_count == 500

    # 1. JSON TrackletMetadata live ingest
    new_meta = {
        "tracklet_id": "cam02_t9001",
        "video_id": "cam02_feed_sample.mp4",
        "cam_id": "Cam 02 (Admin Corridor)",
        "person_track_id": 901,
        "start_sec": 12.0,
        "end_sec": 15.0,
        "crop_paths": ["data/tracklets/crops/cam02_t9001_c1.jpg"],
    }
    resp_json = seeded_client.post("/api/v1/ingest/live", json=new_meta)
    assert resp_json.status_code == 200
    ingest_data = resp_json.json()
    assert ingest_data["status"] == "indexed"
    assert ingest_data["tracklet_id"] == "cam02_t9001"
    assert ingest_data["total_indexed_tracklets"] == initial_count + 1

    # 2. Multipart crop image + metadata JSON live ingest
    img_buf = io.BytesIO()
    Image.new("RGB", (128, 384), color=(200, 40, 40)).save(img_buf, format="JPEG")
    img_buf.seek(0)

    multipart_meta = {
        "tracklet_id": "cam04_t9002",
        "video_id": "cam04_feed_sample.mp4",
        "cam_id": "Cam 04 (Cafeteria)",
        "person_track_id": 902,
        "start_sec": 20.0,
        "end_sec": 23.0,
        "crop_paths": ["data/tracklets/crops/cam04_t9002_c1.jpg"],
    }
    resp_multi = seeded_client.post(
        "/api/v1/ingest/live",
        data={"metadata": json.dumps(multipart_meta)},
        files={"crop": ("cam04_t9002_c1.jpg", img_buf, "image/jpeg")},
    )
    assert resp_multi.status_code == 200
    multi_data = resp_multi.json()
    assert multi_data["total_indexed_tracklets"] == initial_count + 2

    # Verify /health reflects live update without server restart
    updated_health = seeded_client.get("/health").json()
    assert updated_health["total_indexed_tracklets"] == initial_count + 2


# ============================================================================
# Test 5: FAISSIndexManager unit norm, save/load, static mount, and config
# ============================================================================
def test_faiss_manager_normalization_persistence_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert ensure_contracts_on_path().exists()
    assert ensure_dl_core_on_path().exists()
    assert resolve_repo_path(None, "data/raw_videos") == (REPO_ROOT / "data" / "raw_videos").resolve()

    monkeypatch.setenv("CCTV_MOCK_ENCODER", "1")
    settings = get_settings()
    assert settings.mock_encoder is True
    assert settings.embed_dim == EMBED_DIM

    # Zero-norm and non-finite vectors must be rejected
    with pytest.raises(ValueError):
        normalize_embedding(np.zeros((1, EMBED_DIM), dtype=np.float32))
    bad_vec = np.ones((1, EMBED_DIM), dtype=np.float32)
    bad_vec[0, 0] = np.nan
    with pytest.raises(ValueError):
        normalize_embedding(bad_vec)

    manager = FAISSIndexManager()
    raw_vec = np.random.default_rng(123).standard_normal((1, EMBED_DIM)).astype(np.float32) * 5.0
    meta = TrackletMetadata(
        tracklet_id="cam01_t0001",
        video_id="cam01_feed_sample.mp4",
        cam_id="Cam 01 (Main Gate)",
        person_track_id=1,
        start_sec=0.0,
        end_sec=3.0,
        crop_paths=["data/tracklets/crops/cam01_t0001_c1.jpg"],
    )
    assigned = manager.add(raw_vec, meta)
    assert assigned == 0
    assert manager.total_indexed_tracklets == 1

    # Query with the exact same unnormalized direction -> cosine similarity == 1.0
    results = manager.search(raw_vec * 2.0, top_k=1)
    assert len(results) == 1
    assert results[0].similarity_score == pytest.approx(1.0, abs=1e-4)
    assert results[0].tracklet.tracklet_id == "cam01_t0001"

    # Save and reload roundtrip
    idx_file = tmp_path / "test.index"
    man_file = tmp_path / "test_manifest.json"
    manager.save(idx_file, man_file)

    reloaded = FAISSIndexManager()
    loaded_count = reloaded.load(idx_file, man_file)
    assert loaded_count == 1
    assert reloaded.cameras == ["Cam 01 (Main Gate)"]

    # Verify /static/crops static mount serves crop thumbnails
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    thumb = crops_dir / "sample_thumb.jpg"
    Image.new("RGB", (64, 192), color=(20, 120, 220)).save(thumb)
    monkeypatch.setenv("CCTV_CROPS_DIR", str(crops_dir))

    with TestClient(create_app()) as static_client:
        static_resp = static_client.get("/static/crops/sample_thumb.jpg")
        assert static_resp.status_code == 200
        assert len(static_resp.content) > 0
