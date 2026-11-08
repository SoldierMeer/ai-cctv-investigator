"""Unit test suite for Stage 1: Video Ingestion & Tracking Pipeline.

Tests:
1. Bounding box validation and empty frame handling in PersonDetector.
2. Track ID persistence across 10 sequential synthetic frames with ByteTrack.
3. Laplacian variance scoring prefers sharp boxes over blurred boxes.
4. TrackletMetadata Pydantic v2 validation & 384x128 letterbox crop generation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

# Ensure vision pipeline source and shared contracts are in path
VISION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = VISION_ROOT / "src"
SHARED_CONTRACTS = REPO_ROOT / "packages" / "shared-contracts" / "python"

for p in [str(SRC_DIR), str(SHARED_CONTRACTS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from crop_sampler import TrackletCropSampler, letterbox_crop
from detector import PersonDetector
from tracker import PedestrianTracker
from schemas import TrackletMetadata


# ============================================================================
# Test 1: Bounding box validation and empty frame handling in PersonDetector
# ============================================================================
def test_person_detector_empty_and_validation() -> None:
    """Test 1: PersonDetector gracefully handles None, empty, and invalid frames."""
    detector = PersonDetector(weights="yolov8n.pt", conf_thresh=0.45, iou_thresh=0.50)

    # 1. Test None frame
    res_none = detector.detect(None)  # type: ignore[arg-type]
    assert isinstance(res_none, np.ndarray)
    assert res_none.shape == (0, 6)

    # 2. Test empty array frame
    empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
    res_empty = detector.detect(empty_frame)
    assert isinstance(res_empty, np.ndarray)
    assert res_empty.shape == (0, 6)

    # 3. Test 1D invalid array
    res_1d = detector.detect(np.zeros(10, dtype=np.uint8))
    assert res_1d.shape == (0, 6)

    # 4. Mock model prediction to verify strict filtering to COCO Class 0 (person)
    mock_model = MagicMock()
    mock_boxes = MagicMock()

    # Detections:
    # Row 0: Person (class 0.0), conf 0.88, valid box [10, 20, 100, 200]
    # Row 1: Backpack/Car (class 2.0), conf 0.92, should be filtered out
    # Row 2: Person (class 0.0), conf 0.65, valid box [150, 40, 250, 300]
    mock_data = MagicMock()
    mock_data.cpu.return_value.numpy.return_value = np.array(
        [
            [10.0, 20.0, 100.0, 200.0, 0.88, 0.0],
            [30.0, 50.0, 120.0, 180.0, 0.92, 2.0],
            [150.0, 40.0, 250.0, 300.0, 0.65, 0.0],
        ],
        dtype=np.float32,
    )
    mock_boxes.data = mock_data
    mock_boxes.__len__.return_value = 3

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_model.predict.return_value = [mock_result]

    # Inject mock model
    detector._model = mock_model

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)

    assert isinstance(detections, np.ndarray)
    assert detections.shape == (2, 6), "Must strictly filter out non-person class detections"
    # Verify all remaining detections have class_id == 0.0
    assert np.all(detections[:, 5] == 0.0)
    # Verify bounding boxes are valid (x2 > x1, y2 > y1)
    assert np.all(detections[:, 2] > detections[:, 0])
    assert np.all(detections[:, 3] > detections[:, 1])


# ============================================================================
# Test 2: Track ID persistence across 10 sequential synthetic frames
# ============================================================================
def test_track_id_persistence_across_sequential_frames() -> None:
    """Test 2: ByteTrack maintains persistent integer ID across 10 moving frames."""
    tracker = PedestrianTracker(track_thresh=0.45, match_thresh=0.40, lost_buffer=30)

    track_ids_observed = []
    # 10 sequential synthetic frames with a moving white box on black background
    for t in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Synthetic box moves rightward by 15px per frame
        x1 = 50.0 + t * 15.0
        y1 = 100.0
        x2 = 120.0 + t * 15.0
        y2 = 260.0

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 255, 255),
            -1,
        )

        detections = np.array([[x1, y1, x2, y2, 0.95, 0.0]], dtype=np.float32)
        active_tracks = tracker.update(detections, frame)

        assert len(active_tracks) == 1, f"Frame {t}: expected exactly 1 active track"
        track_info = active_tracks[0]
        track_ids_observed.append(track_info["track_id"])

        # Check bbox structure
        bbox = track_info["bbox"]
        assert len(bbox) == 4
        assert bbox[2] > bbox[0]
        assert bbox[3] > bbox[1]

    # Verify ID persistence: the ID must be identical across all 10 frames
    unique_ids = set(track_ids_observed)
    assert len(unique_ids) == 1, f"Track ID changed during continuous track: {track_ids_observed}"
    persistent_id = track_ids_observed[0]

    # Test short occlusion (re-identification across lost buffer)
    # 2 frames of complete occlusion (no detections)
    for _ in range(2):
        empty_tracks = tracker.update(np.empty((0, 6), dtype=np.float32))
        assert len(empty_tracks) == 0

    # Frame 12: Person reappears near projected location
    reappear_x1 = 50.0 + 12 * 15.0
    reappear_x2 = 120.0 + 12 * 15.0
    reappear_det = np.array([[reappear_x1, 100.0, reappear_x2, 260.0, 0.92, 0.0]], dtype=np.float32)
    recovered_tracks = tracker.update(reappear_det)

    assert len(recovered_tracks) == 1
    assert recovered_tracks[0]["track_id"] == persistent_id, "ByteTrack must re-identify persistent ID after occlusion"


# ============================================================================
# Test 3: Laplacian variance scoring prefers sharp boxes over blurred boxes
# ============================================================================
def test_laplacian_sharpness_scoring() -> None:
    """Test 3: Quality metric scores sharp in-focus crops strictly higher than blurred crops."""
    # Create sharp synthetic pedestrian crop with crisp high-frequency edges and patterns
    sharp_crop = np.zeros((300, 150, 3), dtype=np.uint8)
    for y in range(20, 280, 20):
        cv2.line(sharp_crop, (10, y), (140, y), (255, 255, 255), 2)
    cv2.rectangle(sharp_crop, (30, 40), (120, 260), (200, 200, 200), -1)
    cv2.circle(sharp_crop, (75, 80), 25, (0, 0, 255), -1)

    # Artificially blur with Gaussian blur
    blurred_crop = cv2.GaussianBlur(sharp_crop, (31, 31), 10.0)

    sharp_score = TrackletCropSampler.compute_quality_score(sharp_crop)
    blurred_score = TrackletCropSampler.compute_quality_score(blurred_crop)

    assert sharp_score > 0.0, "Sharp crop must have non-zero Laplacian variance"
    assert sharp_score > blurred_score * 2.0, (
        f"Sharp score ({sharp_score:.2f}) must be significantly higher than blurred score ({blurred_score:.2f})"
    )


# ============================================================================
# Test 4: Verify TrackletMetadata matches Pydantic v2 validation & letterbox 384x128
# ============================================================================
def test_tracklet_metadata_pydantic_validation(tmp_path: Path) -> None:
    """Test 4: TrackletCropSampler produces valid TrackletMetadata and 384x128 crops."""
    test_crops_dir = tmp_path / "crops"
    sampler = TrackletCropSampler(crops_dir=test_crops_dir, target_h=384, target_w=128)

    # Construct synthetic frames buffer with 4 frames
    frames_buffer = []
    base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(base_frame, (100, 100), (200, 350), (180, 180, 180), -1)

    timestamps = [0.0, 0.9, 1.8, 2.7]
    for ts in timestamps:
        frame_copy = base_frame.copy()
        # Add slight texture variation
        cv2.circle(frame_copy, (150, int(150 + ts * 20)), 15, (255, 255, 255), -1)
        frames_buffer.append(
            {
                "frame": frame_copy,
                "bbox": [100.0, 100.0, 200.0, 350.0],
                "timestamp": ts,
            }
        )

    metadata = sampler.process_tracklet(
        track_id=7,
        cam_id="Cam 02 (Admin Corridor)",
        video_id="cam02_admin_1080p.mp4",
        frames_buffer=frames_buffer,
    )

    # Pydantic v2 validation
    assert isinstance(metadata, TrackletMetadata)
    assert metadata.person_track_id == 7
    assert metadata.cam_id == "Cam 02 (Admin Corridor)"
    assert metadata.video_id == "cam02_admin_1080p.mp4"
    assert metadata.start_sec >= 0.0
    assert metadata.end_sec > metadata.start_sec
    assert len(metadata.crop_paths) == 3, "Must select exactly top-3 crops"

    # Verify each crop file was written to disk and has exact shape 384x128x3
    for rel_path in metadata.crop_paths:
        filename = Path(rel_path).name
        full_path = test_crops_dir / filename
        assert full_path.exists(), f"Crop file does not exist on disk: {full_path}"

        img = cv2.imread(str(full_path))
        assert img is not None, f"Failed to read written crop: {full_path}"
        # Shape: (height, width, channels) -> (384, 128, 3)
        assert img.shape == (384, 128, 3), f"Crop must have shape (384, 128, 3), got {img.shape}"

    # Verify Pydantic dump and re-validation
    serialized = metadata.model_dump()
    revalidated = TrackletMetadata.model_validate(serialized)
    assert revalidated == metadata
