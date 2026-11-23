"""FastAPI entrypoint for AI CCTV Investigator (Stages 4 & 5A)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import AsyncIterator, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_APP_DIR = str(Path(__file__).resolve().parent)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from core.config import Settings, get_settings  # noqa: E402
from core.encoder import EncoderService  # noqa: E402
from core.faiss_index import FAISSIndexManager  # noqa: E402
from routers.ingest_router import router as ingest_router  # noqa: E402
from routers.search_router import router as search_router  # noqa: E402
from routers.video_router import router as video_router  # noqa: E402


class HealthResponse(BaseModel):
    """Health check payload reporting FAISS index capacity, encoder mode, and cameras."""

    status: str
    total_indexed_tracklets: int
    model_status: str
    cameras: List[str]


def create_app(settings_override: Settings | None = None) -> FastAPI:
    """Create configured FastAPI instance with lifespan state and static mounts."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        settings = settings_override or get_settings()
        settings.crops_dir.mkdir(parents=True, exist_ok=True)
        settings.raw_videos_dir.mkdir(parents=True, exist_ok=True)

        faiss_manager = FAISSIndexManager(
            embed_dim=settings.embed_dim,
            hnsw_m=settings.hnsw_m,
            ef_construction=settings.hnsw_ef_construction,
            ef_search=settings.hnsw_ef_search,
        )
        if settings.faiss_index_path.exists() and settings.tracklet_manifest_path.exists():
            faiss_manager.load(settings.faiss_index_path, settings.tracklet_manifest_path)

        encoder = EncoderService(
            checkpoint_path=settings.irra_checkpoint_path,
            mock_mode=settings.mock_encoder,
        )

        application.state.settings = settings
        application.state.faiss_manager = faiss_manager
        application.state.encoder = encoder
        yield

    init_settings = settings_override or get_settings()
    init_settings.crops_dir.mkdir(parents=True, exist_ok=True)

    application = FastAPI(
        title="AI CCTV Investigator API",
        version="0.1.0",
        description="Stage 4 & 5 semantic pedestrian search, HTTP 206 streaming, and live tracklet ingest.",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=init_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.mount(
        "/static/crops",
        StaticFiles(directory=str(init_settings.crops_dir)),
        name="static-crops",
    )

    application.include_router(search_router)
    application.include_router(video_router)
    application.include_router(ingest_router)

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    def health_check(request: Request) -> HealthResponse:
        faiss_manager: FAISSIndexManager = request.app.state.faiss_manager
        encoder: EncoderService = request.app.state.encoder
        return HealthResponse(
            status="ok",
            total_indexed_tracklets=faiss_manager.total_indexed_tracklets,
            model_status=encoder.model_status,
            cameras=faiss_manager.cameras,
        )

    return application


app = create_app()
