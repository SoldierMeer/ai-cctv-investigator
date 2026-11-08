/**
 * AI CCTV Investigator - Shared Contracts (TypeScript)
 * Single source of truth mirrored from packages/shared-contracts/python
 */

// Pipeline Constants
export const EMBED_DIM: number = 512;
export const TRACKLET_WIN_SEC: number = 3.0;
export const TOP_CROPS_PER_TRACKLET: number = 3;
export const NUM_HORIZONTAL_STRIPES: number = 4;
export const TARGET_LATENCY_MS: number = 400.0;
export const HNSW_M: number = 32;
export const HNSW_EF_CONSTRUCTION: number = 128;
export const HNSW_EF_SEARCH: number = 64;

// Schemas / DTOs
export interface TrackletMetadata {
  tracklet_id: string;
  video_id: string;
  cam_id: string;
  person_track_id: number;
  start_sec: number;
  end_sec: number;
  crop_paths: string[];
}

export interface SearchQueryRequest {
  query: string;
  top_k?: number;
  cam_filter?: string | null;
}

export interface SearchResultItem {
  rank: number;
  similarity_score: number;
  tracklet: TrackletMetadata;
}

export interface SearchResponse {
  query: string;
  latency_ms: number;
  total_indexed_tracklets: number;
  results: SearchResultItem[];
}
