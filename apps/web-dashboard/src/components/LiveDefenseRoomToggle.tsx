"use client";

import React, { useEffect, useState } from "react";
import { Database, Radio, ShieldCheck, WifiOff } from "lucide-react";
import { fetchHealth, type HealthResponse } from "@/lib/apiClient";

export interface LiveDefenseRoomToggleProps {
  liveDefenseMode: boolean;
  onToggle: (enabled: boolean) => void;
  externalTrackletCount?: number | null;
  onHealthUpdate?: (health: HealthResponse) => void;
}

export function LiveDefenseRoomToggle({
  liveDefenseMode,
  onToggle,
  externalTrackletCount = null,
  onHealthUpdate,
}: LiveDefenseRoomToggleProps) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [offline, setOffline] = useState<boolean>(false);

  useEffect(() => {
    let active = true;

    const pollHealth = async () => {
      try {
        const data = await fetchHealth();
        if (!active) return;
        setHealth(data);
        setOffline(false);
        onHealthUpdate?.(data);
      } catch {
        if (!active) return;
        setOffline(true);
      }
    };

    void pollHealth();
    const timer = setInterval(() => {
      void pollHealth();
    }, 5000);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [onHealthUpdate]);

  const indexedCount =
    externalTrackletCount ?? health?.total_indexed_tracklets ?? null;

  return (
    <div className="flex flex-wrap items-center gap-3">
      {/* Live FAISS Index Counter */}
      <div className="inline-flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/90 px-3 py-1.5 text-xs">
        <Database className="h-3.5 w-3.5 text-cyan-400" />
        <span className="text-slate-400">Indexed Tracklets:</span>
        <span className="font-mono font-semibold text-slate-100">
          {indexedCount !== null ? indexedCount.toLocaleString() : "--"}
        </span>
      </div>

      {/* Backend Health / Model Status Pill */}
      <div
        className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 font-mono text-xs ${
          offline
            ? "border-rose-500/40 bg-rose-950/50 text-rose-300"
            : "border-emerald-500/30 bg-emerald-950/40 text-emerald-300"
        }`}
      >
        {offline ? (
          <>
            <WifiOff className="h-3.5 w-3.5" />
            <span>API OFFLINE</span>
          </>
        ) : (
          <>
            <ShieldCheck className="h-3.5 w-3.5" />
            <span>{health ? health.model_status.toUpperCase() : "CONNECTING"}</span>
          </>
        )}
      </div>

      {/* Campus vs. Live Defense Room Mode Toggle */}
      <div
        role="group"
        aria-label="Camera scope selector"
        className="inline-flex rounded-lg border border-slate-800 bg-slate-950 p-0.5 text-xs"
      >
        <button
          type="button"
          onClick={() => onToggle(false)}
          className={`rounded-md px-3 py-1.5 font-medium transition ${
            !liveDefenseMode
              ? "bg-cyan-500/20 text-cyan-200"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          All Campus Cameras
        </button>
        <button
          type="button"
          onClick={() => onToggle(true)}
          className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 font-medium transition ${
            liveDefenseMode
              ? "bg-rose-500/20 text-rose-200"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Radio
            className={`h-3.5 w-3.5 ${
              liveDefenseMode ? "animate-pulse text-rose-400" : "text-slate-500"
            }`}
          />
          <span>Live Defense Room</span>
        </button>
      </div>
    </div>
  );
}
