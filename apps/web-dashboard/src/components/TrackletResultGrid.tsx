"use client";

import React, { useState } from "react";
import { Camera, Clock, Hash, SearchX, User } from "lucide-react";
import type { SearchResultItem, TrackletMetadata } from "@/lib/contracts";
import { getCropImageUrl } from "@/lib/apiClient";
import { formatSimilarityPercent, formatTimestamp } from "@/lib/formatters";

interface CropThumbnailProps {
  cropPath: string;
  trackletId: string;
}

function CropThumbnail({ cropPath, trackletId }: CropThumbnailProps) {
  const [failed, setFailed] = useState(false);

  if (failed || !cropPath) {
    return (
      <div className="flex h-36 w-full flex-col items-center justify-center gap-1.5 rounded-lg border border-slate-800 bg-slate-950/90 px-2 text-center">
        <div className="flex h-11 w-11 items-center justify-center rounded-full border border-slate-700 bg-slate-900 text-slate-400">
          <User className="h-5 w-5" />
        </div>
        <span className="font-mono text-[11px] text-slate-400">{trackletId}</span>
        <span className="text-[10px] text-slate-600">Mock Crop Preview</span>
      </div>
    );
  }

  return (
    <div className="relative h-36 w-full overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
      <img
        src={getCropImageUrl(cropPath)}
        alt={`Tracklet ${trackletId} crop`}
        onError={() => setFailed(true)}
        className="h-full w-full object-contain"
      />
    </div>
  );
}

export interface TrackletResultGridProps {
  results: SearchResultItem[];
  selectedTrackletId: string | null;
  onSelectTracklet: (tracklet: TrackletMetadata) => void;
  isLoading: boolean;
  hasSearched: boolean;
}

export function TrackletResultGrid({
  results,
  selectedTrackletId,
  onSelectTracklet,
  isLoading,
  hasSearched,
}: TrackletResultGridProps) {
  return (
    <section className="rounded-xl border border-slate-800 bg-[#0b1224] p-4 shadow-lg shadow-black/30">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
          Ranked Tracklet Matches
        </h2>
        <span className="font-mono text-xs text-slate-400">
          {results.length} {results.length === 1 ? "match" : "matches"}
        </span>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, idx) => (
            <div
              key={idx}
              className="h-56 animate-pulse rounded-lg border border-slate-800 bg-slate-900/60 p-3"
            >
              <div className="mb-3 h-32 rounded bg-slate-800/80" />
              <div className="mb-2 h-3 w-2/3 rounded bg-slate-800" />
              <div className="h-3 w-1/2 rounded bg-slate-800" />
            </div>
          ))}
        </div>
      ) : results.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-slate-800 bg-slate-950/60 px-6 py-12 text-center">
          <SearchX className="mb-3 h-8 w-8 text-slate-500" />
          <p className="text-sm font-medium text-slate-300">
            {hasSearched
              ? "No matching tracklets found for this query & camera filter."
              : "Enter a suspect description or select a quick query above to search."}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Results are ranked by 512-D cosine similarity in FAISS HNSW.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {results.map((item) => {
            const { tracklet, rank, similarity_score } = item;
            const isSelected = selectedTrackletId === tracklet.tracklet_id;
            const primaryCrop = tracklet.crop_paths[0] ?? "";

            return (
              <button
                key={tracklet.tracklet_id}
                type="button"
                onClick={() => onSelectTracklet(tracklet)}
                className={`group flex flex-col rounded-lg border p-3 text-left transition ${
                  isSelected
                    ? "border-cyan-400 bg-slate-900/95 ring-1 ring-cyan-400/50"
                    : "border-slate-800 bg-slate-900/60 hover:border-slate-600 hover:bg-slate-900"
                }`}
              >
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="inline-flex items-center rounded-md bg-cyan-500/15 px-2 py-0.5 font-mono text-xs font-semibold text-cyan-300">
                    #{rank}
                  </span>
                  <span className="inline-flex items-center rounded-md border border-emerald-500/30 bg-emerald-950/50 px-2 py-0.5 font-mono text-xs font-semibold text-emerald-300">
                    {formatSimilarityPercent(similarity_score)}
                  </span>
                </div>

                <CropThumbnail
                  cropPath={primaryCrop}
                  trackletId={tracklet.tracklet_id}
                />

                <div className="mt-2.5 space-y-1.5">
                  <div className="flex items-center gap-1.5 text-xs font-medium text-slate-200">
                    <Camera className="h-3.5 w-3.5 shrink-0 text-cyan-400" />
                    <span className="truncate">{tracklet.cam_id}</span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                    <span className="inline-flex items-center gap-1">
                      <Clock className="h-3 w-3 text-slate-500" />
                      [{formatTimestamp(tracklet.start_sec)} –{" "}
                      {formatTimestamp(tracklet.end_sec)}]
                    </span>
                    <span className="inline-flex items-center gap-0.5 text-slate-500">
                      <Hash className="h-3 w-3" />
                      {tracklet.person_track_id}
                    </span>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}
