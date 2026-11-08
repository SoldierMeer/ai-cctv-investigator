"""3-Second Windowing and Laplacian Crop Quality Sampler.

Extracts top-3 sharpest crops from pedestrian tracklets and letterboxes them
to 384x128 for downstream Re-ID feature extraction.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure shared contracts are importable
REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_CONTRACTS = REPO_ROOT / "packages" / "shared-contracts" / "python"
if str(SHARED_CONTRACTS) not in sys.path:
    sys.path.insert(0, str(SHARED_CONTRACTS))

try:
    from schemas import TrackletMetadata
    from constants import TRACKLET_WIN_SEC, TOP_CROPS_PER_TRACKLET
except ImportError:
    from pydantic import BaseModel, Field

    TRACKLET_WIN_SEC = 3.0
    TOP_CROPS_PER_TRACKLET = 3

    class TrackletMetadata(BaseModel):  # type: ignore[no-redef]
        tracklet_id: str
        video_id: str
        cam_id: str
        person_track_id: int
        start_sec: float
        end_sec: float
        crop_paths: List[str]


def letterbox_crop(
    crop: np.ndarray,
    target_h: int = 384,
    target_w: int = 128,
    fill_value: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Resize image to target_h x target_w preserving aspect ratio with letterbox padding."""
    if crop is None or crop.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    letterboxed = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=fill_value,
    )
    return letterboxed


class TrackletCropSampler:
    """Extracts, scores, letterboxes, and saves top crops from tracklet frames."""

    def __init__(
        self,
        crops_dir: Optional[Union[str, Path]] = None,
        target_h: int = 384,
        target_w: int = 128,
        win_sec: float = TRACKLET_WIN_SEC,
        top_crops: int = TOP_CROPS_PER_TRACKLET,
    ) -> None:
        if crops_dir is None:
            self.crops_dir = REPO_ROOT / "data" / "tracklets" / "crops"
        else:
            self.crops_dir = Path(crops_dir)

        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.target_h = target_h
        self.target_w = target_w
        self.win_sec = win_sec
        self.top_crops = top_crops

    @staticmethod
    def compute_quality_score(crop: np.ndarray) -> float:
        """Calculate image sharpness score: Laplacian variance * area (w * h)."""
        if crop is None or crop.size == 0:
            return 0.0

        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return 0.0

        if len(crop.shape) == 3 and crop.shape[2] == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        elif len(crop.shape) == 3 and crop.shape[2] == 1:
            gray = crop[:, :, 0]
        else:
            gray = crop

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return lap_var * float(w * h)

    def extract_crop(self, frame: np.ndarray, bbox: Union[List[float], Tuple[float, ...]]) -> np.ndarray:
        """Safely clip bounding box and extract sub-image from frame."""
        if frame is None or frame.size == 0 or len(bbox) < 4:
            return np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)

        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, min(int(round(bbox[0])), frame_w - 1))
        y1 = max(0, min(int(round(bbox[1])), frame_h - 1))
        x2 = max(x1 + 1, min(int(round(bbox[2])), frame_w))
        y2 = max(y1 + 1, min(int(round(bbox[3])), frame_h))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        return crop

    def process_tracklet(
        self,
        track_id: int,
        cam_id: str,
        video_id: str,
        frames_buffer: List[Dict[str, Any]],
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> TrackletMetadata:
        """Process buffered frames of a single tracklet window.

        Args:
            track_id: Persistent integer ID from ByteTrack.
            cam_id: Camera identifier string (e.g. "Cam 01").
            video_id: Video filename.
            frames_buffer: List of dicts, each with at least 'frame' and 'bbox'.
            start_sec: Optional start timestamp in seconds.
            end_sec: Optional end timestamp in seconds.

        Returns:
            Validated TrackletMetadata adhering strictly to shared contracts.
        """
        # Determine timestamps
        timestamps = [
            f.get("timestamp", f.get("time_sec", f.get("sec", None)))
            for f in frames_buffer
            if isinstance(f, dict)
        ]
        valid_ts = [t for t in timestamps if t is not None]

        if start_sec is None:
            start_sec = float(min(valid_ts)) if valid_ts else 0.0
        if end_sec is None:
            end_sec = float(max(valid_ts)) if valid_ts else (start_sec + self.win_sec)
        if end_sec <= start_sec:
            end_sec = start_sec + self.win_sec

        # Generate unique tracklet_id e.g. cam01_t0042
        clean_cam = re.sub(r"[^a-zA-Z0-9]", "", cam_id.lower())
        tracklet_id = f"{clean_cam}_t{track_id:04d}_{int(start_sec * 100):06d}"

        # Score all frames in buffer
        scored_candidates: List[Tuple[float, np.ndarray]] = []
        for item in frames_buffer:
            if not isinstance(item, dict):
                continue
            frame = item.get("frame")
            bbox = item.get("bbox")
            if frame is None or bbox is None:
                continue

            crop = self.extract_crop(frame, bbox)
            score = self.compute_quality_score(crop)
            scored_candidates.append((score, crop))

        # Sort descending by quality score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Select top crops, pad if fewer than self.top_crops
        selected_crops: List[np.ndarray] = []
        if scored_candidates:
            for i in range(self.top_crops):
                idx = min(i, len(scored_candidates) - 1)
                selected_crops.append(scored_candidates[idx][1])
        else:
            # Fallback black crop
            blank = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
            selected_crops = [blank] * self.top_crops

        # Letterbox and save crops to disk
        crop_paths: List[str] = []
        for i, crop in enumerate(selected_crops, start=1):
            letterboxed = letterbox_crop(crop, self.target_h, self.target_w)
            filename = f"{tracklet_id}_c{i}.jpg"
            disk_path = self.crops_dir / filename
            cv2.imwrite(str(disk_path), letterboxed)

            # Store relative path anchored to repo root
            rel_path = f"data/tracklets/crops/{filename}"
            crop_paths.append(rel_path)

        return TrackletMetadata(
            tracklet_id=tracklet_id,
            video_id=video_id,
            cam_id=cam_id,
            person_track_id=track_id,
            start_sec=round(float(start_sec), 2),
            end_sec=round(float(end_sec), 2),
            crop_paths=crop_paths,
        )
