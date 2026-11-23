"use client";

import React from "react";
import { Activity, Camera, Loader2, Search, Sparkles, X } from "lucide-react";
import { TARGET_LATENCY_MS } from "@/lib/contracts";

const SAMPLE_PROMPTS: string[] = [
  "man wearing a maroon hoodie and dark jeans carrying a black backpack",
  "man in a green kurta and black vest carrying a brown shoulder bag",
  "woman in a white jacket and dark trousers near main gate",
  "person wearing a blue windbreaker holding a silver laptop bag",
];

const CAMERA_OPTIONS: { label: string; value: string }[] = [
  { label: "All Cameras", value: "" },
  { label: "Cam 01 (Main Gate)", value: "Cam 01" },
  { label: "Cam 02 (Admin Corridor)", value: "Cam 02" },
  { label: "Cam 03 (Library Walkway)", value: "Cam 03" },
  { label: "Cam 04 (Cafeteria)", value: "Cam 04" },
];

export interface NaturalLanguageSearchBarProps {
  query: string;
  onQueryChange: (value: string) => void;
  camFilter: string;
  onCamFilterChange: (value: string) => void;
  onSearch: (overrideQuery?: string) => void;
  isLoading: boolean;
  latencyMs: number | null;
  liveDefenseMode: boolean;
}

export function NaturalLanguageSearchBar({
  query,
  onQueryChange,
  camFilter,
  onCamFilterChange,
  onSearch,
  isLoading,
  latencyMs,
  liveDefenseMode,
}: NaturalLanguageSearchBarProps) {
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (query.trim().length >= 3 && !isLoading) {
      onSearch();
    }
  };

  const isWithinBudget = latencyMs !== null && latencyMs < TARGET_LATENCY_MS;

  const latencyPillClass =
    latencyMs === null
      ? "border-slate-700 bg-slate-900/80 text-slate-400"
      : isWithinBudget
        ? "border-emerald-500/40 bg-emerald-950/60 text-emerald-300"
        : "border-amber-500/40 bg-amber-950/60 text-amber-300";

  const latencyDotClass =
    latencyMs === null
      ? "bg-slate-500"
      : isWithinBudget
        ? "bg-emerald-400"
        : "bg-amber-400";

  return (
    <section className="rounded-xl border border-slate-800 bg-[#0b1224] p-4 shadow-lg shadow-black/30">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-cyan-400" />
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
            Cross-Modal Suspect Query
          </h2>
        </div>

        <div
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-mono ${latencyPillClass}`}
          title={`End-to-end SLA target: < ${TARGET_LATENCY_MS} ms`}
        >
          <span className={`h-2 w-2 rounded-full ${latencyDotClass}`} />
          <Activity className="h-3.5 w-3.5" />
          <span>
            {latencyMs !== null ? `${latencyMs.toFixed(1)} ms` : "-- ms"}
          </span>
          {latencyMs !== null && (
            <span className="text-[10px] uppercase opacity-80">
              {isWithinBudget ? "SLA OK" : "SLOW"}
            </span>
          )}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="Describe suspect clothing, accessories, or appearance (min 3 chars)..."
            aria-label="Natural language suspect description"
            minLength={3}
            className="w-full rounded-lg border border-slate-700 bg-slate-950/90 py-2.5 pl-10 pr-9 text-sm text-slate-100 placeholder-slate-500 outline-none transition focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/40"
          />
          {query.length > 0 && (
            <button
              type="button"
              onClick={() => onQueryChange("")}
              aria-label="Clear search query"
              className="absolute right-2.5 top-1/2 -translate-y-1/2 rounded p-1 text-slate-400 transition hover:bg-slate-800 hover:text-slate-200"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>

        <div className="relative sm:w-56">
          <Camera className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <select
            value={liveDefenseMode ? "Live Defense Cam" : camFilter}
            onChange={(e) => onCamFilterChange(e.target.value)}
            disabled={liveDefenseMode}
            aria-label="Filter by camera"
            className="w-full appearance-none rounded-lg border border-slate-700 bg-slate-950/90 py-2.5 pl-9 pr-8 text-sm text-slate-200 outline-none transition focus:border-cyan-500 disabled:cursor-not-allowed disabled:border-rose-500/40 disabled:text-rose-300"
          >
            {liveDefenseMode ? (
              <option value="Live Defense Cam">Live Defense Cam</option>
            ) : (
              CAMERA_OPTIONS.map((opt) => (
                <option key={opt.label} value={opt.value}>
                  {opt.label}
                </option>
              ))
            )}
          </select>
        </div>

        <button
          type="submit"
          disabled={isLoading || query.trim().length < 3}
          className="inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-500 px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Searching</span>
            </>
          ) : (
            <>
              <Search className="h-4 w-4" />
              <span>Investigate</span>
            </>
          )}
        </button>
      </form>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
          Quick Queries:
        </span>
        {SAMPLE_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            type="button"
            disabled={isLoading}
            onClick={() => {
              onQueryChange(prompt);
              onSearch(prompt);
            }}
            className="rounded-md border border-slate-800 bg-slate-900/90 px-2.5 py-1 text-left text-xs text-slate-300 transition hover:border-cyan-500/50 hover:bg-slate-800 hover:text-cyan-200 disabled:opacity-50"
          >
            {prompt}
          </button>
        ))}
      </div>
    </section>
  );
}
