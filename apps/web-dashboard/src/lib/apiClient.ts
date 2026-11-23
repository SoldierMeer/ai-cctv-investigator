import type { SearchQueryRequest, SearchResponse } from "@/lib/contracts";

export interface HealthResponse {
  status: string;
  total_indexed_tracklets: number;
  model_status: string;
  cameras: string[];
}

export const API_BASE_URL: string = (
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
).replace(/\/+$/, "");

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Health check failed (${response.status} ${response.statusText})`);
  }

  return (await response.json()) as HealthResponse;
}

export async function searchTracklets(req: SearchQueryRequest): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(req),
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errBody = (await response.json()) as { detail?: string };
      if (errBody?.detail) {
        detail = typeof errBody.detail === "string" ? errBody.detail : JSON.stringify(errBody.detail);
      }
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(`Search failed (${response.status}): ${detail}`);
  }

  return (await response.json()) as SearchResponse;
}

export function getVideoStreamUrl(videoId: string): string {
  const cleanId = videoId.trim();
  return `${API_BASE_URL}/api/v1/video/${encodeURIComponent(cleanId)}`;
}

export function getCropImageUrl(cropPath: string): string {
  const normalized = cropPath.replace(/\\/g, "/").trim();
  if (/^https?:\/\//i.test(normalized)) {
    return normalized;
  }
  if (normalized.startsWith("/static/crops/")) {
    return `${API_BASE_URL}${normalized}`;
  }
  const prefix = "data/tracklets/crops/";
  const idx = normalized.indexOf(prefix);
  const relativeName =
    idx >= 0
      ? normalized.slice(idx + prefix.length)
      : normalized.split("/").filter(Boolean).pop() || normalized;
  const encodedPath = relativeName
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  return `${API_BASE_URL}/static/crops/${encodedPath}`;
}
