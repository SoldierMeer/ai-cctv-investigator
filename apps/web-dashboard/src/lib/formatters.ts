/**
 * Formatting helpers shared across AI CCTV Investigator dashboard components.
 */

export function formatTimestamp(seconds: number): string {
  const safeSec = Math.max(0, Math.floor(seconds));
  const hrs = Math.floor(safeSec / 3600);
  const mins = Math.floor((safeSec % 3600) / 60);
  const secs = safeSec % 60;
  if (hrs > 0) {
    return `${String(hrs).padStart(2, "0")}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

export function formatSimilarityPercent(score: number): string {
  const clamped = Math.max(0, Math.min(1, score));
  return `${(clamped * 100).toFixed(1)}%`;
}

export function formatCameraShortName(camId: string): string {
  const match = camId.match(/^([^(]+)/);
  return match ? match[1].trim() : camId;
}
