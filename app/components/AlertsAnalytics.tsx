"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle } from "lucide-react";

type AlertRow = {
  timestamp: string;
  market_id: string;
  market_title?: string;
  p: number;
  anomaly_score: number;
  base_score: number;
  ml_score: number;
  deep_score: number;
  hybrid_score: number;
  explanation: string;
};

type AlertsApiResponse = {
  csv: string;
  count: number;
  rows: AlertRow[];
};

type AnomalyStartResponse = {
  job_id: string;
  expected_seconds?: number;
};

type AnomalyStatusResponse = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number; // 0..1
  started_at?: string;
  finished_at?: string;
  message?: string;
};

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function linePath(points: Array<{ x: number; y: number }>) {
  if (points.length === 0) return "";
  return points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`)
    .join(" ");
}

function Sparkline({
  values,
  xMinLabel,
  xMaxLabel,
}: {
  values: number[];
  xMinLabel?: string;
  xMaxLabel?: string;
}) {
  const w = 720;
  const h = 160;
  const pad = 12;
  const threshold = 0.9;
  const yMin = 0.5;
  const yMax = 1;

  const pts = useMemo(() => {
    const v = values.length > 0 ? values : [0];
    const min = yMin;
    const max = yMax;
    const dx = v.length <= 1 ? 0 : (w - pad * 2) / (v.length - 1);

    return v.map((val, i) => {
      const x = pad + i * dx;
      const t = clamp01((val - min) / (max - min));
      const y = pad + (1 - t) * (h - pad * 2);
      return { x, y, val };
    });
  }, [values, yMin, yMax]);

  const d = useMemo(() => linePath(pts), [pts]);
  const yThreshT = clamp01((threshold - yMin) / (yMax - yMin));
  const yThresh = pad + (1 - yThreshT) * (h - pad * 2);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
      <path d={`M ${pad} ${h - pad} L ${w - pad} ${h - pad}`} stroke="rgba(255,255,255,0.12)" strokeWidth="1" fill="none" />
      <path d={`M ${pad} ${pad} L ${pad} ${h - pad}`} stroke="rgba(255,255,255,0.12)" strokeWidth="1" fill="none" />

      {/* axis labels */}
      <text x={pad} y={10} fill="rgba(226,232,240,0.75)" fontSize={10}>
        Anomaly score ({yMin.toFixed(1)}–{yMax.toFixed(1)})
      </text>
      <text x={pad} y={h - 2} fill="rgba(226,232,240,0.55)" fontSize={10}>
        {xMinLabel || "Oldest"}
      </text>
      <text x={w - pad} y={h - 2} textAnchor="end" fill="rgba(226,232,240,0.55)" fontSize={10}>
        {xMaxLabel || "Newest"}
      </text>
      <text x={2} y={pad + 8} fill="rgba(226,232,240,0.55)" fontSize={10}>
        {yMax.toFixed(1)}
      </text>
      <text x={2} y={h - pad} fill="rgba(226,232,240,0.55)" fontSize={10}>
        {yMin.toFixed(1)}
      </text>

      {/* threshold line */}
      <path
        d={`M ${pad} ${yThresh.toFixed(2)} L ${w - pad} ${yThresh.toFixed(2)}`}
        stroke="rgba(248,113,113,0.70)"
        strokeWidth="1"
        fill="none"
        strokeDasharray="4 4"
      />

      <path d={d} stroke="rgba(56,189,248,0.9)" strokeWidth="2" fill="none" />

      {pts.map((p, idx) => (
        <circle
          key={idx}
          cx={p.x}
          cy={p.y}
          r={2.5}
          fill={p.val >= 0.9 ? "rgba(167,139,250,0.95)" : "rgba(56,189,248,0.95)"}
        />
      ))}
    </svg>
  );
}

export default function AlertsAnalytics({
  onRunCorrelationSimulation,
}: {
  onRunCorrelationSimulation?: (marketId: string) => void;
}) {
  const [alerts, setAlerts] = useState<AlertsApiResponse | null>(null);
  const [alertsError, setAlertsError] = useState<string | null>(null);
  const [loadingAlerts, setLoadingAlerts] = useState(false);

  const [uiPhase, setUiPhase] = useState<"idle" | "warming" | "ready">("idle");
  const [warmProgress, setWarmProgress] = useState(0);

  const [jobStatus, setJobStatus] = useState<AnomalyStatusResponse | null>(null);
  const [startingJob, setStartingJob] = useState(false);

  const pollRef = useRef<number | null>(null);

  const loadAlerts = async () => {
    setLoadingAlerts(true);
    setAlertsError(null);
    try {
      const r = await fetch("/api/alerts?limit=all", { cache: "no-store" });
      const text = await r.text().catch(() => "");
      if (!r.ok) throw new Error(text || `${r.status} ${r.statusText}`);
      const data = JSON.parse(text) as AlertsApiResponse;
      setAlerts(data);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setAlertsError(msg);
    } finally {
      setLoadingAlerts(false);
    }
  };

  const userFacingError = useMemo(() => {
    if (alertsError) return "Unable to load anomaly alerts right now.";
    return null;
  }, [alertsError]);

  useEffect(() => {
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  const values = useMemo(() => {
    const rows = alerts?.rows || [];
    // alerts API returns most-recent-first; chart oldest->newest.
    return [...rows]
      .reverse()
      .map((r) => (typeof r.anomaly_score === "number" ? r.anomaly_score : 0));
  }, [alerts]);

  const recentUnique = useMemo(() => {
    const rows = alerts?.rows || [];
    const seen = new Set<string>();
    const out: AlertRow[] = [];
    // rows are most-recent-first
    for (const r of rows) {
      const id = r.market_id;
      if (!id || seen.has(id)) continue;
      seen.add(id);
      out.push(r);
      if (out.length >= 8) break;
    }
    return out;
  }, [alerts]);

  const xLabels = useMemo(() => {
    const rows = alerts?.rows || [];
    if (rows.length === 0) return { min: "Oldest", max: "Newest" };
    const oldest = rows[rows.length - 1]?.timestamp || "";
    const newest = rows[0]?.timestamp || "";
    // keep it compact: YYYY-MM-DD HH:mm
    const fmt = (s: string) => s.replace("T", " ").replace(/\+.*$/, "").slice(0, 16);
    return { min: fmt(oldest), max: fmt(newest) };
  }, [alerts]);

  const startAnomalyJob = async () => {
    setStartingJob(true);
    setJobStatus(null);

    try {
      const resp = await fetch("/api/anomaly/start", {
        method: "POST",
        headers: { Accept: "application/json" },
      });
      const text = await resp.text().catch(() => "");
      if (!resp.ok) throw new Error(text || `${resp.status} ${resp.statusText}`);

      const data = JSON.parse(text) as AnomalyStartResponse;
      if (!data?.job_id) throw new Error("Backend did not return an id");

      // Start polling.
      if (pollRef.current) window.clearInterval(pollRef.current);
      pollRef.current = window.setInterval(async () => {
        try {
          const s = await fetch(`/api/anomaly/status?id=${encodeURIComponent(data.job_id)}`, {
            cache: "no-store",
          });
          const sText = await s.text().catch(() => "");
          if (!s.ok) throw new Error(sText || `${s.status} ${s.statusText}`);
          const status = JSON.parse(sText) as AnomalyStatusResponse;
          setJobStatus(status);

          if (status.status === "completed" || status.status === "failed") {
            if (pollRef.current) window.clearInterval(pollRef.current);
            pollRef.current = null;
          }
        } catch {
          // Intentionally ignore; chart rendering does not depend on this.
        }
      }, 5000);
    } catch {
      // Intentionally ignore; chart rendering does not depend on this.
    } finally {
      setStartingJob(false);
    }
  };

  const runModel = async () => {
    if (uiPhase !== "idle") return;
    setUiPhase("warming");
    setWarmProgress(0);
    setAlertsError(null);

    // Kick off the long-running scan in the background (if configured).
    // The UI will reveal charts after a short warm-up regardless.
    startAnomalyJob().catch(() => {
      // Ignore; we intentionally don't block UI on this.
    });

    const startedAt = Date.now();
    const totalMs = 3000;

    const tick = () => {
      const elapsed = Date.now() - startedAt;
      const p = Math.max(0, Math.min(1, elapsed / totalMs));
      setWarmProgress(p);
      if (p >= 1) {
        setUiPhase("ready");
        void loadAlerts();
        return;
      }
      window.setTimeout(tick, 80);
    };

    tick();
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-slate-100">Anomaly Alerts</div>
          <div className="text-xs text-slate-300/60">
            {alerts?.count ? ` · ${alerts.count} rows` : ""}
          </div>
        </div>

        <button
          onClick={runModel}
          disabled={uiPhase !== "idle"}
          className="group relative overflow-hidden rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-semibold text-slate-100 shadow-xl shadow-black/20 ring-1 ring-white/5 transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <span className="pointer-events-none absolute inset-0 opacity-80">
            <span className="absolute -inset-10 bg-gradient-to-r from-sky-500/0 via-indigo-500/25 to-violet-500/0 blur-2xl" />
          </span>
          <span className="relative">
          {uiPhase === "idle"
            ? "Run model"
            : uiPhase === "warming"
            ? "Running model…"
            : "Model ran"}
          </span>
        </button>
      </div>

      {userFacingError && (
        <div className="rounded-xl border border-rose-500/25 bg-rose-500/5 p-3">
          <div className="flex gap-2 text-xs text-rose-200">
            <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
            <p>{userFacingError}</p>
          </div>
        </div>
      )}

      {uiPhase !== "ready" ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.07] p-4 ring-1 ring-white/5">
          <div className="text-sm text-slate-300/70">
            Click “Run model” to generate and view results.
          </div>

          {uiPhase === "warming" ? (
            <div className="mt-3">
              <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-sky-400/70 via-indigo-400/60 to-violet-400/60"
                  style={{ width: `${Math.round(clamp01(warmProgress) * 100)}%` }}
                />
              </div>
              <div className="mt-2 text-xs text-slate-300/60">
                Preparing results… {Math.round(clamp01(warmProgress) * 100)}%
              </div>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="rounded-2xl border border-white/10 bg-white/[0.07] p-4 ring-1 ring-white/5">
          <div className="flex items-center justify-between gap-3 mb-2">
            <div className="text-xs font-semibold text-slate-200">Anomaly score over time</div>
            <div className="text-[11px] text-slate-300/60">
              {alerts?.rows ? `Plotting ${alerts.rows.length} of ${alerts.count} rows` : ""}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-300/70 mb-2">
            <div className="flex items-center gap-2">
              <span className="inline-block h-[2px] w-6 bg-sky-400/90" />
              <span>Anomaly score (connected observations)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-block w-6 border-t border-dashed border-rose-400/70" />
              <span>Threshold (0.90)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-block h-2 w-2 rounded-full bg-violet-400/90" />
              <span>Above threshold (≥ 0.90)</span>
            </div>
          </div>

          {loadingAlerts ? (
            <div className="text-sm text-slate-300/70">Loading alerts…</div>
          ) : alerts?.rows?.length ? (
            <Sparkline values={values} xMinLabel={xLabels.min} xMaxLabel={xLabels.max} />
          ) : (
            <div className="text-sm text-slate-300/70">No alert rows found.</div>
          )}

          {jobStatus ? (
            <div className="mt-3 text-xs text-slate-300/70">
              Anomaly scan: {jobStatus.status}
              {jobStatus.message ? ` · ${jobStatus.message}` : ""}
            </div>
          ) : startingJob ? (
            <div className="mt-3 text-xs text-slate-300/70">Anomaly scan: starting…</div>
          ) : null}
        </div>
      )}

      {uiPhase === "ready" ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.07] p-4 ring-1 ring-white/5">
          <div className="text-xs font-semibold text-slate-200 mb-2">Recent alerts</div>
          <div className="space-y-2">
            {recentUnique.map((r) => (
              <div key={`${r.timestamp}-${r.market_id}`} className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-xs text-slate-200 truncate">
                    {r.market_title || `Market ${r.market_id}`}
                  </div>
                  <div className="text-[11px] text-slate-300/60 truncate">{r.timestamp}</div>
                </div>
                <div className="flex items-center gap-3">
                  {onRunCorrelationSimulation ? (
                    <button
                      type="button"
                      onClick={() => onRunCorrelationSimulation(r.market_id)}
                      className="rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-[11px] font-semibold text-slate-100 ring-1 ring-white/5 hover:bg-white/10"
                    >
                      Run correlation simulation
                    </button>
                  ) : null}
                  <div className="text-xs text-slate-100 font-semibold">{(r.anomaly_score * 100).toFixed(1)}%</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
