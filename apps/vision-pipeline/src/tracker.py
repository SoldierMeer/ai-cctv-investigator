"""ByteTrack Pedestrian Tracker for CCTV video streams.

Wraps ByteTrack logic to maintain persistent integer track IDs across
consecutive frames and re-identify pedestrians across short occlusions
(lost track buffer = 30 frames).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


@dataclass
class STrack:
    """Single tracked pedestrian state."""

    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    conf: float
    state: str = "tracked"  # 'tracked' or 'lost'
    lost_frames: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    def predict(self) -> np.ndarray:
        """Predict next bounding box using velocity model."""
        pred_box = self.bbox + self.velocity
        return pred_box

    def update(self, new_bbox: np.ndarray, new_conf: float) -> None:
        """Update track with newly matched detection."""
        # Update velocity with momentum
        delta = new_bbox - self.bbox
        self.velocity = 0.7 * self.velocity + 0.3 * delta
        self.bbox = new_bbox.astype(np.float32)
        self.conf = float(new_conf)
        self.state = "tracked"
        self.lost_frames = 0

    def mark_lost(self) -> None:
        """Mark track as lost and advance predicted location."""
        self.state = "lost"
        self.lost_frames += 1
        self.bbox = self.predict()


class PedestrianTracker:
    """ByteTrack-based pedestrian tracker maintaining persistent IDs."""

    def __init__(
        self,
        track_thresh: float = 0.45,
        match_thresh: float = 0.50,
        low_conf_thresh: float = 0.10,
        lost_buffer: int = 30,
        frame_rate: int = 30,
    ) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.low_conf_thresh = low_conf_thresh
        self.lost_buffer = lost_buffer
        self.frame_rate = frame_rate

        self._tracks: List[STrack] = []
        self._next_id: int = 1

    def _associate(
        self,
        tracks: List[STrack],
        detections: np.ndarray,
        iou_threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match tracks with detections using IoU distance."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Build IoU cost matrix
        num_t = len(tracks)
        num_d = len(detections)
        iou_matrix = np.zeros((num_t, num_d), dtype=np.float32)
        for t_idx, track in enumerate(tracks):
            pred_box = track.predict()
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = compute_iou(pred_box, det[:4])

        matched_pairs: List[Tuple[int, int]] = []
        matched_tracks = set()
        matched_dets = set()

        # Greedy bipartite matching by descending IoU
        flat_indices = np.argsort(-iou_matrix, axis=None)
        for flat_idx in flat_indices:
            t_idx = int(flat_idx // num_d)
            d_idx = int(flat_idx % num_d)
            score = iou_matrix[t_idx, d_idx]

            if score < iou_threshold:
                break
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue

            matched_pairs.append((t_idx, d_idx))
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

        unmatched_tracks = [t for t in range(num_t) if t not in matched_tracks]
        unmatched_dets = [d for d in range(num_d) if d not in matched_dets]

        return matched_pairs, unmatched_tracks, unmatched_dets

    def update(self, detections: np.ndarray, frame: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Update tracker with frame detections.

        Args:
            detections: np.ndarray of shape (N, 6) [x1, y1, x2, y2, conf, cls]
                        or (N, 5) [x1, y1, x2, y2, conf].
            frame: Optional video frame (H, W, 3) for optional appearance features.

        Returns:
            List of active tracks: [{"track_id": int, "bbox": [x1, y1, x2, y2], "conf": float}]
        """
        if detections is None or len(detections) == 0:
            # Advance all existing tracks as lost
            surviving_tracks = []
            for track in self._tracks:
                track.mark_lost()
                if track.lost_frames <= self.lost_buffer:
                    surviving_tracks.append(track)
            self._tracks = surviving_tracks
            return []

        det_arr = np.asarray(detections, dtype=np.float32)
        if len(det_arr.shape) == 1:
            det_arr = np.expand_dims(det_arr, axis=0)

        confs = det_arr[:, 4] if det_arr.shape[1] >= 5 else np.ones(len(det_arr), dtype=np.float32)

        # 1. Split detections into high-score and low-score (ByteTrack 2-stage association)
        high_mask = confs >= self.track_thresh
        low_mask = (confs >= self.low_conf_thresh) & (~high_mask)

        dets_high = det_arr[high_mask]
        dets_low = det_arr[low_mask]

        # 2. First association with high-confidence detections
        matched_1, unmatched_tracks_1_idx, unmatched_dets_1_idx = self._associate(
            self._tracks, dets_high, self.match_thresh
        )

        for t_idx, d_idx in matched_1:
            det = dets_high[d_idx]
            self._tracks[t_idx].update(det[:4], det[4] if len(det) >= 5 else 1.0)

        # 3. Second association: remaining unmatched tracks with low-confidence detections
        unmatched_tracks_1 = [self._tracks[i] for i in unmatched_tracks_1_idx]
        matched_2, unmatched_tracks_2_idx, _ = self._associate(
            unmatched_tracks_1, dets_low, self.match_thresh * 0.8
        )

        for t_sub_idx, d_idx in matched_2:
            det = dets_low[d_idx]
            unmatched_tracks_1[t_sub_idx].update(det[:4], det[4] if len(det) >= 5 else 1.0)

        # 4. Handle still unmatched tracks -> mark as lost or drop if beyond lost_buffer
        for t_sub_idx in unmatched_tracks_2_idx:
            unmatched_tracks_1[t_sub_idx].mark_lost()

        # Clean up tracks exceeding lost_buffer
        self._tracks = [t for t in self._tracks if t.lost_frames <= self.lost_buffer]

        # 5. Initialize new tracks from unmatched high-confidence detections
        for d_idx in unmatched_dets_1_idx:
            det = dets_high[d_idx]
            new_track = STrack(
                track_id=self._next_id,
                bbox=det[:4].astype(np.float32),
                conf=float(det[4]) if len(det) >= 5 else 1.0,
                state="tracked",
                lost_frames=0,
            )
            self._next_id += 1
            self._tracks.append(new_track)

        # 6. Return currently active tracks
        active_results: List[Dict[str, Any]] = []
        for track in self._tracks:
            if track.state == "tracked":
                active_results.append(
                    {
                        "track_id": int(track.track_id),
                        "bbox": [round(float(c), 2) for c in track.bbox],
                        "conf": round(float(track.conf), 4),
                    }
                )

        return active_results
