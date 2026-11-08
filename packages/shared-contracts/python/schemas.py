from typing import List, Optional
from pydantic import BaseModel, Field


class TrackletMetadata(BaseModel):
    tracklet_id: str = Field(..., description="Unique ID e.g. cam02_t0042")
    video_id: str = Field(..., description="Source video filename")
    cam_id: str = Field(..., description="Camera identifier e.g. Cam 02")
    person_track_id: int = Field(..., description="ByteTrack persistent integer ID")
    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., gt=0.0)
    crop_paths: List[str] = Field(..., min_length=1, max_length=3)


class SearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language person description")
    top_k: int = Field(default=10, ge=1, le=50)
    cam_filter: Optional[str] = Field(default=None)


class SearchResultItem(BaseModel):
    rank: int
    similarity_score: float
    tracklet: TrackletMetadata


class SearchResponse(BaseModel):
    query: str
    latency_ms: float
    total_indexed_tracklets: int
    results: List[SearchResultItem]
