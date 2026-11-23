"""Live Defense-Room Webcam Daemon for real-time Stage 1 + Stage 5 live ingestion.

Captures frames from a local webcam (`--source 0`), RTSP stream, or video file,
runs `PersonDetector` + `PedestrianTracker`, groups detections into 3.0-second windows,
extracts top-3 sharpest crops via `TrackletCropSampler` tagged `cam_id="Live Defense Cam"`,
and uploads multipart payloads to `POST /api/v1/ingest/live`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import httpx
import numpy as np

VISION_SRC = Path(__file__).resolve().parent
REPO_ROOT = VISION_SRC.parents[2]
SHARED_CONTRACTS_PY = REPO_ROOT / "packages" / "shared-contracts" / "python"

for _p in (str(VISION_SRC), str(SHARED_CONTRACTS_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from constants import TRACKLET_WIN_SEC  # noqa: E402
from crop_sampler import TrackletCropSampler  # noqa: E402
from detector import PersonDetector, SyntheticBoxDetector  # noqa: E402
from schemas import TrackletMetadata  # noqa: E402
from tracker import PedestrianTracker  # noqa: E402

logger = logging.getLogger(__name__)


def generate_synthetic_frame(
    frame_idx: int, width: int = 640, height: int = 480
) -> np.ndarray:
    """Create a synthetic frame with a textured moving pedestrian box for `--dry-run-frames`."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    x1 = 80 + (frame_idx * 10) % max(100, width - 220)
    y1 = 90
    x2 = x1 + 90
    y2 = 360

    # Head, torso, legs, and backpack regions for crisp Laplacian texture
    cv2.rectangle(frame, (x1 + 25, y1), (x1 + 65, y1 + 45), (210, 185, 165), -1)
    cv2.rectangle(frame, (x1 + 10, y1 + 50), (x2 - 10, y1 + 165), (50, 90, 210), -1)
    cv2.rectangle(frame, (x1 + 15, y1 + 170), (x2 - 15, y2), (160, 160, 175), -1)
    cv2.rectangle(frame, (x2 - 22, y1 + 65), (x2, y1 + 155), (220, 220, 220), 2)
    return frame


def _resolve_crop_files(crop_paths: Sequence[str], crops_dir: Path) -> List[Path]:
    resolved: List[Path] = []
    for rel in crop_paths:
        p_repo = REPO_ROOT / rel
        if p_repo.exists():
            resolved.append(p_repo)
        else:
            p_local = crops_dir / Path(rel).name
            if p_local.exists():
                resolved.append(p_local)
    return resolved


def post_tracklet_to_api(
    api_url: str,
    metadata: TrackletMetadata,
    crops_dir: Path,
    http_client: Optional[Any] = None,
    timeout_sec: float = 5.0,
) -> Dict[str, Any]:
    """POST multipart crop image(s) + JSON `TrackletMetadata` to `/api/v1/ingest/live`."""
    crop_files = _resolve_crop_files(metadata.crop_paths, crops_dir)
    files_payload: List[Tuple[str, Tuple[str, bytes, str]]] = [
        ("crop", (p.name, p.read_bytes(), "image/jpeg")) for p in crop_files
    ]
    form_data = {"metadata": metadata.model_dump_json()}

    endpoint = "/api/v1/ingest/live"
    if http_client is not None:
        response = http_client.post(endpoint, data=form_data, files=files_payload or None)
    else:
        url = f"{api_url.rstrip('/')}{endpoint}"
        response = httpx.post(
            url, data=form_data, files=files_payload or None, timeout=timeout_sec
        )

    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


def _resolve_detector(
    detector: Optional[Any],
    mock_detector: bool,
    dry_run_frames: Optional[int],
    device: str,
) -> Any:
    if detector is not None:
        return detector
    if mock_detector:
        return SyntheticBoxDetector()
    if dry_run_frames is not None and not (REPO_ROOT / "yolov8n.pt").exists():
        logger.info(
            "Using SyntheticBoxDetector for offline --dry-run-frames (yolov8n.pt not present locally)."
        )
        return SyntheticBoxDetector()
    return PersonDetector(device=device)


def run_live_daemon(
    source: str = "0",
    api_url: str = "http://localhost:8000",
    cam_id: str = "Live Defense Cam",
    dry_run_frames: Optional[int] = None,
    crops_dir: Union[str, Path] = "data/tracklets/crops",
    fps: float = 30.0,
    device: str = "cpu",
    mock_detector: bool = False,
    skip_upload: bool = False,
    detector: Optional[Any] = None,
    http_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run live webcam / stream tracklet extraction and upload to FastAPI."""
    crp_dir = Path(crops_dir)
    if not crp_dir.is_absolute():
        crp_dir = REPO_ROOT / crp_dir

    det = _resolve_detector(detector, mock_detector, dry_run_frames, device)
    tracker = PedestrianTracker(frame_rate=int(round(fps)))
    sampler = TrackletCropSampler(crops_dir=crp_dir, win_sec=TRACKLET_WIN_SEC)

    cap: Optional[cv2.VideoCapture] = None
    use_synthetic = False
    video_label = f"webcam_{source}"

    if dry_run_frames is not None:
        source_path = Path(source)
        if not source.isdigit() and source_path.exists():
            cap = cv2.VideoCapture(str(source_path))
            video_label = source_path.name
            if not cap.isOpened():
                cap = None
                use_synthetic = True
        else:
            use_synthetic = True
            video_label = "live_defense_dry_run.mp4"
    else:
        src_arg: Union[int, str] = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(src_arg)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        stream_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if np.isfinite(stream_fps) and stream_fps > 0.0:
            fps = stream_fps

    buffers: Dict[int, Dict[str, Any]] = {}
    emitted: List[TrackletMetadata] = []
    uploaded_count = 0
    frames_processed = 0

    def _emit_tracklet(tid: int, buf: Dict[str, Any], end_ts: float) -> None:
        nonlocal uploaded_count
        if not buf["frames"]:
            return
        meta = sampler.process_tracklet(
            track_id=tid,
            cam_id=cam_id,
            video_id=video_label,
            frames_buffer=buf["frames"],
            start_sec=buf["win_start"],
            end_sec=max(buf["win_start"] + 0.1, end_ts),
        )
        emitted.append(meta)

        if skip_upload:
            return

        try:
            post_tracklet_to_api(
                api_url=api_url,
                metadata=meta,
                crops_dir=crp_dir,
                http_client=http_client,
            )
            uploaded_count += 1
        except httpx.HTTPError as exc:
            if dry_run_frames is not None and http_client is None:
                logger.warning(
                    "[DRY-RUN] API server unreachable at %s (%s); skipping network upload.",
                    api_url,
                    exc,
                )
            else:
                logger.error("Failed to upload tracklet %s: %s", meta.tracklet_id, exc)

    t0_wall = time.monotonic()
    try:
        while True:
            if dry_run_frames is not None and frames_processed >= dry_run_frames:
                break

            if use_synthetic:
                frame = generate_synthetic_frame(frames_processed)
            else:
                assert cap is not None
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

            timestamp = (
                frames_processed / fps
                if dry_run_frames is not None
                else (time.monotonic() - t0_wall)
            )
            detections = det.detect(frame)
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

                if (timestamp - buf["win_start"]) >= TRACKLET_WIN_SEC:
                    _emit_tracklet(tid, buf, buf["win_start"] + TRACKLET_WIN_SEC)
                    buffers[tid] = {"win_start": timestamp, "frames": []}

            frames_processed += 1
    except KeyboardInterrupt:
        logger.info("Live webcam daemon stopped by user interrupt.")
    finally:
        if cap is not None:
            cap.release()

    for tid, buf in buffers.items():
        if buf["frames"]:
            _emit_tracklet(tid, buf, buf["frames"][-1]["timestamp"])

    return {
        "frames_processed": frames_processed,
        "tracklets_emitted": len(emitted),
        "uploaded_count": uploaded_count,
        "tracklets": emitted,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the live defense-room daemon."""
    parser = argparse.ArgumentParser(
        description="Live Defense-Room Webcam Daemon streaming 3s tracklets to FastAPI."
    )
    parser.add_argument("--source", type=str, default="0", help="Camera index (e.g. 0) or RTSP/MP4 path.")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    parser.add_argument("--cam-id", type=str, default="Live Defense Cam")
    parser.add_argument("--crops-dir", type=str, default="data/tracklets/crops")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dry-run-frames",
        type=int,
        default=None,
        help="Run N synthetic/offline frames and exit cleanly without requiring a physical webcam.",
    )
    parser.add_argument("--mock-detector", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)
    summary = run_live_daemon(
        source=args.source,
        api_url=args.api_url,
        cam_id=args.cam_id,
        dry_run_frames=args.dry_run_frames,
        crops_dir=args.crops_dir,
        fps=args.fps,
        device=args.device,
        mock_detector=args.mock_detector,
        skip_upload=args.skip_upload,
    )
    print(
        f"Live daemon finished: frames={summary['frames_processed']}, "
        f"tracklets_emitted={summary['tracklets_emitted']}, "
        f"uploaded={summary['uploaded_count']}"
    )


if __name__ == "__main__":
    main()
