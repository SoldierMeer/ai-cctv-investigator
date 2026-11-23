"use client";

import React, { useCallback, useMemo, useState } from "react";
import { AlertCircle, ShieldAlert, X } from "lucide-react";
import { CrossCameraTransitMap } from "@/components/CrossCameraTransitMap";
import { LiveDefenseRoomToggle } from "@/components/LiveDefenseRoomToggle";
import { NaturalLanguageSearchBar } from "@/components/NaturalLanguageSearchBar";
import { TrackletResultGrid } from "@/components/TrackletResultGrid";
import { VideoPlaybackHUD } from "@/components/VideoPlaybackHUD";
import { searchTracklets, type HealthResponse } from "@/lib/apiClient";
import type { SearchResultItem, TrackletMetadata } from "@/lib/contracts";

export default function InvestigatorCommandCenterPage() {
  const [query, setQuery] = useState<string>("");
  const [camFilter, setCamFilter] = useState<string>("");
  const [liveDefenseMode, setLiveDefenseMode] = useState<boolean>(false);
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [selectedTracklet, setSelectedTracklet] =
    useState<TrackletMetadata | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [totalIndexed, setTotalIndexed] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [hasSearched, setHasSearched] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const executeSearch = useCallback(
    async (overrideQuery?: string, overrideDefenseMode?: boolean) => {
      const effectiveQuery = (overrideQuery ?? query).trim();
      if (effectiveQuery.length < 3) return;

      const isDefense = overrideDefenseMode ?? liveDefenseMode;
      const effectiveCamFilter = isDefense
        ? "Live Defense Cam"
        : camFilter.trim() || null;

      setIsLoading(true);
      setErrorMsg(null);
      setHasSearched(true);

      try {
        const res = await searchTracklets({
          query: effectiveQuery,
          top_k: 10,
          cam_filter: effectiveCamFilter,
        });
        setResults(res.results);
        setLatencyMs(res.latency_ms);
        setTotalIndexed(res.total_indexed_tracklets);
        setSelectedTracklet(res.results[0]?.tracklet ?? null);
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Failed to query API server.";
        setErrorMsg(msg);
      } finally {
        setIsLoading(false);
      }
    },
    [query, camFilter, liveDefenseMode]
  );

  const handleToggleDefenseMode = useCallback(
    (enabled: boolean) => {
      setLiveDefenseMode(enabled);
      if (hasSearched && query.trim().length >= 3) {
        void executeSearch(query, enabled);
      }
    },
    [executeSearch, hasSearched, query]
  );

  const handleHealthUpdate = useCallback((health: HealthResponse) => {
    setTotalIndexed(health.total_indexed_tracklets);
  }, []);

  const selectedMatch = useMemo(() => {
    if (!selectedTracklet) return null;
    return (
      results.find(
        (item) => item.tracklet.tracklet_id === selectedTracklet.tracklet_id
      ) ?? null
    );
  }, [results, selectedTracklet]);

  return (
    <div className="min-h-screen bg-[#040814] text-slate-100">
      {/* Command Center Top Navigation Bar */}
      <header className="sticky top-0 z-20 border-b border-slate-800/90 bg-[#070d1d]/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-cyan-500/40 bg-cyan-500/10 text-cyan-400">
              <ShieldAlert className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-slate-100 sm:text-lg">
                AI CCTV INVESTIGATOR
              </h1>
              <p className="font-mono text-[11px] uppercase tracking-wider text-slate-400">
                CLIP ViT-B/16 · IRRA 4-Stripe · FAISS HNSW Command Center
              </p>
            </div>
          </div>

          <LiveDefenseRoomToggle
            liveDefenseMode={liveDefenseMode}
            onToggle={handleToggleDefenseMode}
            externalTrackletCount={totalIndexed}
            onHealthUpdate={handleHealthUpdate}
          />
        </div>
      </header>

      {/* Main 2-Column Command Workspace */}
      <main className="mx-auto max-w-[1600px] px-4 py-6 sm:px-6">
        {errorMsg && (
          <div className="mb-5 flex items-center justify-between gap-3 rounded-lg border border-rose-500/40 bg-rose-950/40 px-4 py-3 text-sm text-rose-200">
            <div className="flex items-center gap-2.5">
              <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
              <span>{errorMsg}</span>
            </div>
            <button
              type="button"
              onClick={() => setErrorMsg(null)}
              aria-label="Dismiss error"
              className="rounded p-1 text-rose-300 transition hover:bg-rose-900/50"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          {/* Left Column: Search Controls + Ranked Tracklets */}
          <div className="space-y-6 lg:col-span-7">
            <NaturalLanguageSearchBar
              query={query}
              onQueryChange={setQuery}
              camFilter={camFilter}
              onCamFilterChange={setCamFilter}
              onSearch={(overrideQ) => {
                void executeSearch(overrideQ);
              }}
              isLoading={isLoading}
              latencyMs={latencyMs}
              liveDefenseMode={liveDefenseMode}
            />

            <TrackletResultGrid
              results={results}
              selectedTrackletId={selectedTracklet?.tracklet_id ?? null}
              onSelectTracklet={setSelectedTracklet}
              isLoading={isLoading}
              hasSearched={hasSearched}
            />
          </div>

          {/* Right Column: HTTP 206 Video HUD + Chronological Transit Map */}
          <div className="space-y-6 lg:col-span-5">
            <VideoPlaybackHUD
              tracklet={selectedTracklet}
              rank={selectedMatch?.rank ?? null}
              similarityScore={selectedMatch?.similarity_score ?? null}
            />

            <CrossCameraTransitMap
              results={results}
              selectedTrackletId={selectedTracklet?.tracklet_id ?? null}
              onSelectTracklet={setSelectedTracklet}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
