"""Search router exposing `POST /api/v1/search` backed by FAISS IndexHNSWFlat."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from core.config import ensure_contracts_on_path
from core.encoder import EncoderService
from core.faiss_index import FAISSIndexManager

ensure_contracts_on_path()
from schemas import SearchQueryRequest, SearchResponse  # noqa: E402

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search_tracklets(payload: SearchQueryRequest, request: Request) -> SearchResponse:
    """Encode natural language query and return top-K tracklets ranked by cosine similarity."""
    t_start = time.perf_counter()

    faiss_manager: FAISSIndexManager = request.app.state.faiss_manager
    encoder: EncoderService = request.app.state.encoder

    try:
        query_vec = encoder.encode_text_query(payload.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results = faiss_manager.search(
        query_vector=query_vec,
        top_k=payload.top_k,
        cam_filter=payload.cam_filter,
    )
    latency_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

    return SearchResponse(
        query=payload.query,
        latency_ms=latency_ms,
        total_indexed_tracklets=faiss_manager.total_indexed_tracklets,
        results=results,
    )
