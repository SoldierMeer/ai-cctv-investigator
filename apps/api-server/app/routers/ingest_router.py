"""Live tracklet ingestion router updating FAISS IndexHNSWFlat dynamically."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError
from starlette.datastructures import UploadFile

from core.config import ensure_contracts_on_path
from core.encoder import EncoderService
from core.faiss_index import FAISSIndexManager

ensure_contracts_on_path()
from schemas import TrackletMetadata  # noqa: E402

router = APIRouter(prefix="/api/v1", tags=["ingest"])


class LiveIngestResponse(BaseModel):
    """Response returned after indexing a live tracklet into FAISS."""

    status: str
    tracklet_id: str
    assigned_index_id: int
    total_indexed_tracklets: int


@router.post("/ingest/live", response_model=LiveIngestResponse)
async def ingest_live_tracklet(request: Request) -> LiveIngestResponse:
    """Ingest a tracklet via JSON `TrackletMetadata` or multipart crop image + metadata JSON."""
    faiss_manager: FAISSIndexManager = request.app.state.faiss_manager
    encoder: EncoderService = request.app.state.encoder
    content_type = request.headers.get("content-type", "").lower()

    crop_inputs: List[Union[str, bytes]] = []
    meta_obj: TrackletMetadata

    if "multipart/form-data" in content_type:
        form = await request.form()
        raw_meta = form.get("metadata")
        if raw_meta is None:
            raise HTTPException(
                status_code=422, detail="Multipart payload must include 'metadata' JSON field."
            )
        try:
            meta_str = raw_meta if isinstance(raw_meta, str) else (await raw_meta.read()).decode("utf-8")
            meta_obj = TrackletMetadata.model_validate_json(meta_str)
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid metadata JSON: {exc}") from exc

        for _, value in form.multi_items():
            if isinstance(value, UploadFile):
                file_bytes = await value.read()
                if file_bytes:
                    crop_inputs.append(file_bytes)

        if not crop_inputs:
            crop_inputs = list(meta_obj.crop_paths)
    else:
        try:
            raw_json: Dict[str, Any] = await request.json()
            meta_obj = TrackletMetadata.model_validate(raw_json)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc

        crop_inputs = list(meta_obj.crop_paths)

    embedding = encoder.encode_crops(crop_inputs, fallback_seed=meta_obj.tracklet_id)
    assigned_id = faiss_manager.add(embedding, meta_obj)

    return LiveIngestResponse(
        status="indexed",
        tracklet_id=meta_obj.tracklet_id,
        assigned_index_id=assigned_id,
        total_indexed_tracklets=faiss_manager.total_indexed_tracklets,
    )
