"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Film,
  Play,
  Repeat,
  RotateCcw,
} from "lucide-react";
import type { TrackletMetadata } from "@/lib/contracts";
import { getVideoStreamUrl } from "@/lib/apiClient";
import { formatSimilarityPercent, formatTimestamp } from "@/lib/formatters";

export interface VideoPlaybackHUDProps {
  tracklet: TrackletMetadata | null;
  rank?: number | null;
  similarityScore?: number | null;
}

export function VideoPlaybackHUD({
  tracklet,
  rank = null,
  similarityScore = null,
}: VideoPlaybackHUDProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [loopWindow, setLoopWindow] = useState<boolean>(true);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0);

  const streamUrl = tracklet ? getVideoStreamUrl(tracklet.video_id) : "";

  const seekToWindowStart = (attemptPlay = false) => {
    const vid = videoRef.current;
    if (!vid || !tracklet) return;
    try {
      vid.currentTime = tracklet.start_sec;
      setCurrentTime(tracklet.start_sec);
      if (attemptPlay) {
        void vid.play().catch(() => {
          // Autoplay may be blocked by browser policy; controls remain available
        });
      }
    } catch {
      // Ignore seek errors before metadata is ready
    }
  };

  useEffect(() => {
    setStreamError(null);
    if (!tracklet) {
      setCurrentTime(0);
      return;
    }
    setCurrentTime(tracklet.start_sec);
    const vid = videoRef.current;
    if (vid && vid.readyState >= 1) {
      seekToWindowStart(true);
    }
  }, [tracklet?.tracklet_id, tracklet?.video_id, tracklet?.start_sec]);

  const handleLoadedMetadata = () => {
    seekToWindowStart(true);
  };

  const handleTimeUpdate = () => {
    const vid = videoRef.current;
    if (!vid || !tracklet) return;
    setCurrentTime(vid.currentTime);
    if (loopWindow && vid.currentTime >= tracklet.end_sec) {
      vid.currentTime = tracklet.start_sec;
    }
  };

  const windowSpan = tracklet
    ? Math.max(0.1, tracklet.end_sec - tracklet.start_sec)
    : 3.0;
  const windowProgress = tracklet
    ? Math.max(
        0,
        Math.min(100, ((currentTime - tracklet.start_sec) / windowSpan) * 100)
      )
    : 0;

  return (
    <section className="rounded-xl border border-slate-800 bg-[#0b1224] p-4 shadow-lg shadow-black/30">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Film className="h-4 w-4 text-cyan-400" />
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
            Video Playback HUD
          </h2>
        </div>

        {tracklet && (
          <div className="flex items-center gap-2">
            {rank !== null && (
              <span className="rounded bg-cyan-500/15 px-2 py-0.5 font-mono text-xs font-semibold text-cyan-300">
                Match #{rank}
              </span>
            )}
            {similarityScore !== null && (
              <span className="rounded border border-emerald-500/30 bg-emerald-950/50 px-2 py-0.5 font-mono text-xs text-emerald-300">
                {formatSimilarityPercent(similarityScore)}
              </span>
            )}
          </div>
        )}
      </div>

      {!tracklet ? (
        <div className="flex aspect-video flex-col items-center justify-center rounded-lg border border-dashed border-slate-800 bg-slate-950/70 p-6 text-center">
          <Play className="mb-2 h-8 w-8 text-slate-600" />
          <p className="text-sm font-medium text-slate-300">
            Select a tracklet card to stream CCTV footage
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Uses HTTP 206 byte-range seeking to jump straight to [start_sec, end_sec].
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {streamError ? (
            <div className="flex aspect-video flex-col items-center justify-center rounded-lg border border-amber-500/30 bg-slate-950/90 p-5 text-center">
              <AlertTriangle className="mb-2 h-7 w-7 text-amber-400" />
              <p className="text-sm font-semibold text-amber-200">
                Video File Unavailable in Demo Store
              </p>
              <p className="mt-1 max-w-md text-xs text-slate-400">
                {streamError}
              </p>
              <div className="mt-3 rounded border border-slate-800 bg-slate-900/90 px-3 py-1.5 font-mono text-[11px] text-cyan-300">
                GET /api/v1/video/{tracklet.video_id} · Target Seek:{" "}
                {tracklet.start_sec.toFixed(1)}s – {tracklet.end_sec.toFixed(1)}s
              </div>
            </div>
          ) : (
            <div className="overflow-hidden rounded-lg border border-slate-800 bg-black">
              <video
                ref={videoRef}
                key={streamUrl}
                src={streamUrl}
                controls
                preload="metadata"
                onLoadedMetadata={handleLoadedMetadata}
                onTimeUpdate={handleTimeUpdate}
                onError={() =>
                  setStreamError(
                    `Could not load '${tracklet.video_id}' from ${streamUrl}. Add the MP4 under data/raw_videos/ to enable HTTP 206 streaming.`
                  )
                }
                className="aspect-video w-full bg-black object-contain"
              />
            </div>
          )}

          {/* Highlighted 3-Second Segment Telemetry Bar */}
          <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-3">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs">
              <div>
                <span className="font-semibold text-slate-200">
                  {tracklet.cam_id}
                </span>
                <span className="ml-2 font-mono text-slate-400">
                  ({tracklet.tracklet_id})
                </span>
              </div>
              <span className="font-mono text-cyan-300">
                Window [{formatTimestamp(tracklet.start_sec)} –{" "}
                {formatTimestamp(tracklet.end_sec)}] ({tracklet.start_sec.toFixed(1)}s–
                {tracklet.end_sec.toFixed(1)}s)
              </span>
            </div>

            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full rounded-full bg-cyan-400 transition-all duration-150"
                style={{ width: `${windowProgress}%` }}
              />
            </div>

            <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2">
              <button
                type="button"
                onClick={() => seekToWindowStart(true)}
                className="inline-flex items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1 text-xs font-medium text-slate-200 transition hover:border-cyan-500/60 hover:text-cyan-200"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                <span>Jump to {tracklet.start_sec.toFixed(1)}s</span>
              </button>

              <button
                type="button"
                onClick={() => setLoopWindow((prev) => !prev)}
                aria-pressed={loopWindow}
                className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition ${
                  loopWindow
                    ? "border-cyan-500/50 bg-cyan-500/15 text-cyan-200"
                    : "border-slate-700 bg-slate-900 text-slate-400 hover:text-slate-200"
                }`}
              >
                <Repeat className="h-3.5 w-3.5" />
                <span>Loop 3s Segment: {loopWindow ? "ON" : "OFF"}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
