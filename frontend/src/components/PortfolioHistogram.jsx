import React, { useMemo, useState } from "react";
import { simulateHistogram } from "../mc";

export default function PortfolioHistogram({ positions }) {
  const [trials, setTrials] = useState(200000);
  const [binCount, setBinCount] = useState(30);
  const [maxMultipleEV, setMaxMultipleEV] = useState(5);

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");

  const maxProb = useMemo(() => {
    if (!result?.bins?.length) return 0;
    return Math.max(...result.bins.map(b => b.probability), result.overflowProb || 0);
  }, [result]);

  async function run() {
    setErr("");
    setRunning(true);
    setProgress({ done: 0, total: trials });

    try {
      // Run in a timeout so UI updates before heavy work starts
      await new Promise(r => setTimeout(r, 0));

      const out = simulateHistogram({
        positions,
        trials,
        binCount,
        maxMultipleEV,
        onProgress: (done, total) => setProgress({ done, total }),
      });

      setResult(out);
    } catch (e) {
      setErr(String(e?.message || e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
        <div style={{ fontWeight: 700 }}>Simulation</div>
        <button
          onClick={run}
          disabled={running || positions.length === 0}
          className="button"
        >
          {running ? "Running..." : "Run simulation"}
        </button>
      </div>

      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 10 }}>
        <LabeledNumber label="Trials" value={trials} setValue={setTrials} min={10000} step={10000} />
        <LabeledNumber label="Bins" value={binCount} setValue={setBinCount} min={10} step={1} />
        <LabeledNumber label="Max (×EV)" value={maxMultipleEV} setValue={setMaxMultipleEV} min={1} step={1} />
      </div>

      {running && progress.total > 0 && (
        <div className="subtle" style={{ marginBottom: 8 }}>
          Progress: {progress.done.toLocaleString()} / {progress.total.toLocaleString()}
        </div>
      )}

      {err && <pre style={{ color: "rgba(244,63,94,0.95)", whiteSpace: "pre-wrap" }}>{err}</pre>}

      {!result ? (
        <div className="subtle" style={{ fontSize: 14 }}>
          Run a simulation to see the probability distribution of terminal payout (assumes markets independent).
        </div>
      ) : (
        <>
          <div style={{ fontSize: 14, marginBottom: 10, opacity: 0.85 }}>
            EV ≈ <strong>${result.ev.toFixed(2)}</strong>, histogram range: 0 →{" "}
            <strong>${result.maxValue.toFixed(2)}</strong> (plus overflow)
          </div>

          <HistogramBars bins={result.bins} overflowProb={result.overflowProb} maxProb={maxProb} />
        </>
      )}
    </div>
  );
}

function LabeledNumber({ label, value, setValue, min, step }) {
  return (
    <label style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 13, opacity: 0.9 }}>
      <span className="subtle">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        step={step}
        onChange={(e) => setValue(Number(e.target.value))}
        className="input"
        style={{ width: 130, padding: "8px 10px" }}
      />
    </label>
  );
}

function HistogramBars({ bins, overflowProb, maxProb }) {
  return (
    <div style={{ display: "grid", gap: 6 }}>
      {bins.map((b, i) => {
        const w = maxProb > 0 ? (b.probability / maxProb) * 100 : 0;
        return (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "140px 1fr 80px", gap: 8, alignItems: "center" }}>
            <div style={{ fontSize: 12, opacity: 0.8 }}>
              ${b.range[0].toFixed(0)}–{b.range[1].toFixed(0)}
            </div>
            <div className="barTrack">
              <div className="barFill" style={{ width: `${w}%` }} />
            </div>
            <div style={{ fontSize: 12, textAlign: "right", opacity: 0.9 }}>
              {(b.probability * 100).toFixed(2)}%
            </div>
          </div>
        );
      })}

      {/* Overflow bin */}
      <div style={{ display: "grid", gridTemplateColumns: "140px 1fr 80px", gap: 8, alignItems: "center", marginTop: 6 }}>
        <div style={{ fontSize: 12, opacity: 0.8 }}>≥ max</div>
        <div className="barTrack">
          <div
            className="barFill"
            style={{
              width: `${maxProb > 0 ? (overflowProb / maxProb) * 100 : 0}%`,
            }}
          />
        </div>
        <div style={{ fontSize: 12, textAlign: "right", opacity: 0.9 }}>
          {(overflowProb * 100).toFixed(2)}%
        </div>
      </div>
    </div>
  );
}
