"""HTTP 206 Partial Content byte-range video streaming router."""

from __future__ import annotations

import mimetypes
from pathlib import Path
import re
from typing import AsyncIterator, Tuple

import aiofiles
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/v1", tags=["video"])

_CHUNK_SIZE = 64 * 1024
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def parse_byte_range(range_header: str, file_size: int) -> Tuple[int, int]:
    """Parse HTTP `Range: bytes=start-end` header into inclusive `(start, end)` bounds."""
    if file_size <= 0:
        raise HTTPException(
            status_code=416,
            detail="Cannot satisfy byte range on empty file.",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    match = _RANGE_RE.match(range_header.strip())
    if not match:
        raise HTTPException(
            status_code=416,
            detail=f"Invalid Range header format: {range_header}",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    start_str, end_str = match.group(1), match.group(2)
    if not start_str and not end_str:
        raise HTTPException(
            status_code=416,
            detail="Empty byte range bounds.",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    if start_str:
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1
    else:
        suffix_len = int(end_str)
        if suffix_len <= 0:
            raise HTTPException(
                status_code=416,
                detail="Invalid suffix byte range.",
                headers={"Content-Range": f"bytes */{file_size}"},
            )
        start = max(0, file_size - suffix_len)
        end = file_size - 1

    end = min(end, file_size - 1)
    if start < 0 or start > end or start >= file_size:
        raise HTTPException(
            status_code=416,
            detail="Requested Range Not Satisfiable.",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    return start, end


def resolve_safe_video_path(raw_videos_dir: Path, video_id: str) -> Path:
    """Validate `video_id` against path traversal and resolve inside `raw_videos_dir`."""
    if not video_id or ".." in video_id or "/" in video_id or "\\" in video_id or "\x00" in video_id:
        raise HTTPException(status_code=400, detail="Invalid video_id: path traversal blocked.")

    base_dir = raw_videos_dir.resolve()
    candidate = (base_dir / video_id).resolve()
    if not candidate.is_relative_to(base_dir) or candidate.parent != base_dir:
        raise HTTPException(status_code=400, detail="Invalid video_id: path traversal blocked.")

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")

    return candidate


async def _stream_file_range(path: Path, start: int, end: int) -> AsyncIterator[bytes]:
    remaining = (end - start) + 1
    async with aiofiles.open(path, "rb") as f:
        await f.seek(start)
        while remaining > 0:
            chunk = await f.read(min(_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@router.get("/video/{video_id}")
async def stream_video(video_id: str, request: Request) -> StreamingResponse:
    """Stream CCTV footage with HTTP 206 Partial Content byte-range seeking support."""
    settings = request.app.state.settings
    video_path = resolve_safe_video_path(settings.raw_videos_dir, video_id)
    file_size = video_path.stat().st_size
    media_type, _ = mimetypes.guess_type(video_path.name)
    media_type = media_type or "video/mp4"

    range_header = request.headers.get("range")
    if range_header:
        start, end = parse_byte_range(range_header, file_size)
        content_length = (end - start) + 1
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
        }
        return StreamingResponse(
            _stream_file_range(video_path, start, end),
            status_code=206,
            media_type=media_type,
            headers=headers,
        )

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
    }
    return StreamingResponse(
        _stream_file_range(video_path, 0, max(0, file_size - 1)),
        status_code=200,
        media_type=media_type,
        headers=headers,
    )
