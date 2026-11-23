"""YOLOv8 Person Detector for CCTV streams.

Filters strictly for COCO Class 0 (pedestrians/persons).
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment, misc]


class PersonDetector:
    """YOLOv8-based person detector.

    Filters detections strictly to COCO Class 0 ('person') with configurable
    confidence and IoU thresholds.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf_thresh: float = 0.45,
        iou_thresh: float = 0.50,
        device: str = "cpu",
    ) -> None:
        self.weights = weights
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self._model: Optional[Any] = None

    @property
    def model(self) -> Any:
        """Lazy load YOLO model to defer weight loading until first inference."""
        if self._model is None:
            if YOLO is None:
                raise ImportError(
                    "ultralytics is required for PersonDetector inference. "
                    "Install with 'pip install ultralytics'."
                )
            self._model = YOLO(self.weights)
        return self._model

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect persons in a video frame.

        Args:
            frame: Video frame as numpy array (H, W, 3).

        Returns:
            np.ndarray of shape (N, 6) with columns:
            [x1, y1, x2, y2, confidence, class_id]
            Returns empty array of shape (0, 6) if no persons are detected.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) < 2:
            return np.empty((0, 6), dtype=np.float32)

        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=[0],  # COCO Class 0 = person
            device=self.device,
            verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        data = results[0].boxes.data.cpu().numpy()
        if len(data.shape) != 2 or data.shape[1] < 6:
            return np.empty((0, 6), dtype=np.float32)

        # Strict filtering for Class 0
        person_mask = data[:, 5] == 0.0
        filtered = data[person_mask][:, :6]
        return np.asarray(filtered, dtype=np.float32)


class SyntheticBoxDetector:
    """Lightweight threshold-based detector for offline dry-runs and synthetic video tests."""

    def __init__(self, min_area: int = 400, threshold: int = 35) -> None:
        self.min_area = int(min_area)
        self.threshold = int(threshold)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect bright synthetic pedestrian bounding boxes in `(N, 6)` format."""
        import cv2

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) < 2:
            return np.empty((0, 6), dtype=np.float32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= self.min_area:
                boxes.append([float(x), float(y), float(x + w), float(y + h), 0.95, 0.0])

        if not boxes:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(boxes, dtype=np.float32)

