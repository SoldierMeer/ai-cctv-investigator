"use client";

import React from "react";
import { ArrowRight, MapPin, Route } from "lucide-react";
import type { SearchResultItem, TrackletMetadata } from "@/lib/contracts";
import {
  formatCameraShortName,
  formatSimilarityPercent,
  formatTimestamp,
} from "@/lib/formatters";

export interface CrossCameraTransitMapProps {
  results: SearchResultItem[];
  selectedTrackletId: string | null;
  onSelectTracklet: (tracklet: TrackletMetadata) => void;
}

export function CrossCameraTransitMap({
  results,
  selectedTrackletId,
  onSelectTracklet,
}: CrossCameraTransitMapProps) {
  const chronological = [...results].sort((a, b) => {
    if (a.tracklet.start_sec !== b.tracklet.start_sec) {
      return a.tracklet.start_sec - b.tracklet.start_sec;
    }
    return a.rank - b.rank;
  });

  return (
    <section className="rounded-xl border border-slate-800 bg-[#0b1224] p-4 shadow-lg shadow-black/30">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Route className="h-4 w-4 text-cyan-400" />
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
            Cross-Camera Transit Timeline
          </h2>
        </div>
        <span className="font-mono text-xs text-slate-400">
          {chronological.length} hops
        </span>
      </div>

      {chronological.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-800 bg-slate-950/60 px-4 py-8 text-center text-xs text-slate-500">
          Run a search to reconstruct the suspect&apos;s chronological movement path across campus cameras.
        </div>
      ) : (
        <div className="space-y-4">
          {/* Compact Horizontal Breadcrumb Path */}
          <div className="flex flex-wrap items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-950/75 p-2.5 text-xs">
            {chronological.map((item, idx) => {
              const active = selectedTrackletId === item.tracklet.tracklet_id;
              return (
                <React.Fragment key={`hop-${item.tracklet.tracklet_id}`}>
                  <button
                    type="button"
                    onClick={() => onSelectTracklet(item.tracklet)}
                    className={`rounded px-2 py-0.5 font-mono transition ${
                      active
                        ? "bg-cyan-500/20 text-cyan-200 ring-1 ring-cyan-400/50"
                        : "bg-slate-900 text-slate-300 hover:bg-slate-800"
                    }`}
                  >
                    {formatCameraShortName(item.tracklet.cam_id)} (
                    {formatTimestamp(item.tracklet.start_sec)})
                  </button>
                  {idx < chronological.length - 1 && (
                    <ArrowRight className="h-3.5 w-3.5 shrink-0 text-slate-600" />
                  )}
                </React.Fragment>
              );
            })}
          </div>

          {/* Detailed Chronological Step Timeline */}
          <ol className="relative ml-2 space-y-2.5 border-l border-slate-800 pl-4">
            {chronological.map((item) => {
              const { tracklet, rank, similarity_score } = item;
              const active = selectedTrackletId === tracklet.tracklet_id;

              return (
                <li key={`step-${tracklet.tracklet_id}`} className="relative">
                  <span
                    className={`absolute -left-[21px] top-2.5 h-2.5 w-2.5 rounded-full border-2 ${
                      active
                        ? "border-cyan-300 bg-cyan-400"
                        : "border-slate-600 bg-slate-900"
                    }`}
                  />
                  <button
                    type="button"
                    onClick={() => onSelectTracklet(tracklet)}
                    className={`flex w-full items-center justify-between gap-2 rounded-lg border px-3 py-2 text-left transition ${
                      active
                        ? "border-cyan-500/60 bg-slate-900/95"
                        : "border-slate-800/80 bg-slate-950/60 hover:border-slate-700 hover:bg-slate-900/70"
                    }`}
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5 text-xs font-medium text-slate-200">
                        <MapPin className="h-3.5 w-3.5 shrink-0 text-cyan-400" />
                        <span className="truncate">{tracklet.cam_id}</span>
                      </div>
                      <div className="mt-0.5 font-mono text-[11px] text-slate-400">
                        {formatTimestamp(tracklet.start_sec)} –{" "}
                        {formatTimestamp(tracklet.end_sec)} · {tracklet.tracklet_id}
                      </div>
                    </div>

                    <div className="flex shrink-0 items-center gap-1.5 font-mono text-xs">
                      <span className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-300">
                        #{rank}
                      </span>
                      <span className="rounded bg-emerald-950/70 px-1.5 py-0.5 text-emerald-300">
                        {formatSimilarityPercent(similarity_score)}
                      </span>
                    </div>
                  </button>
                </li>
              );
            })}
          </ol>
        </div>
      )}
    </section>
  );
}
