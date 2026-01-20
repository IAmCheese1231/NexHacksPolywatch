import React, { useMemo, useState } from "react";
import { prettyProb } from "../utils";
import { adjustPositionShares } from "../api";

function positionEV(p) {
  const shares = Number(p.shares);
  const prob = Number(p.implied_probability);
  if (!Number.isFinite(shares) || !Number.isFinite(prob) || prob < 0 || prob > 1) return null;
  return shares * prob;
}

function positionVar(p) {
  const shares = Number(p.shares);
  const prob = Number(p.implied_probability);
  if (!Number.isFinite(shares) || !Number.isFinite(prob) || prob < 0 || prob > 1) return null;
  return (shares * shares) * prob * (1 - prob);
}

export default function PortfolioTable({ positions }) {
  const [busyKey, setBusyKey] = useState(null);
  const [amountByKey, setAmountByKey] = useState({});
  const [err, setErr] = useState("");

  const rows = useMemo(() => Array.isArray(positions) ? positions : [], [positions]);

  async function adjust(p, sign) {
    setErr("");
    const key = `${p.event_slug}:${p.market_slug}:${p.outcome_index}`;
    const amtRaw = amountByKey[key];
    const amt = Number(amtRaw ?? 10);
    if (!(amt > 0)) {
      setErr("Adjustment amount must be > 0.");
      return;
    }

    setBusyKey(`${key}:${sign}`);
    try {
      await adjustPositionShares({
        event_slug: p.event_slug,
        market_slug: p.market_slug,
        outcome_index: Number(p.outcome_index),
        delta_shares: sign * amt,
      });

      // Re-fetch via parent (App) by forcing a reload.
      // Simpler MVP approach: hard reload the embedded dashboard.
      window.location.reload();
    } catch (e) {
      setErr(String(e?.message || e));
    } finally {
      setBusyKey(null);
    }
  }

  return (
    <div>
      {rows.length === 0 ? (
        <div className="subtle">No positions yet.</div>
      ) : (
        <table className="table">
          <thead>
            <tr style={{ textAlign: "left" }}>
              <th>Market</th>
              <th>Outcome</th>
              <th>Shares</th>
              <th>Adjust</th>
              <th>Implied prob</th>
              <th>E[payout]</th>
              <th>Var(payout)</th>
            </tr>
          </thead>

          <tbody>
            {rows.map((p, idx) => {
              const ev = positionEV(p);
              const vr = positionVar(p);
              const key = `${p.event_slug}:${p.market_slug}:${p.outcome_index}`;
              const step = Number(amountByKey[key] ?? 10);
              const minusBusy = busyKey === `${key}:-1`;
              const plusBusy = busyKey === `${key}:1`;

              return (
                <tr key={idx}>
                  <td>
                    {p.market_title}
                    <div className="code">{p.market_slug}</div>
                  </td>

                  <td>{p.outcome_label}</td>

                  <td>{p.shares}</td>

                  <td>
                    <div className="row" style={{ gap: 8 }}>
                      <button
                        className="button buttonMinus"
                        disabled={minusBusy || plusBusy}
                        onClick={() => adjust(p, -1)}
                        style={{ padding: "8px 10px" }}
                        title="Subtract shares"
                      >
                        −
                      </button>

                      <input
                        className="input"
                        type="number"
                        min={0}
                        step="any"
                        value={Number.isFinite(step) ? step : 10}
                        onChange={(e) => {
                          const v = Number(e.target.value);
                          setAmountByKey((prev) => ({
                            ...prev,
                            [key]: Number.isFinite(v) ? v : 0,
                          }));
                        }}
                        style={{ width: 110, padding: "8px 10px" }}
                        aria-label="Adjust shares amount"
                      />

                      <button
                        className="button buttonPlus"
                        disabled={minusBusy || plusBusy}
                        onClick={() => adjust(p, 1)}
                        style={{ padding: "8px 10px" }}
                        title="Add shares"
                      >
                        +
                      </button>
                    </div>
                  </td>

                  <td>{prettyProb(p.implied_probability)}</td>

                  <td>{ev === null ? "—" : `$${ev.toFixed(2)}`}</td>

                  <td>{vr === null ? "—" : vr.toFixed(2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}

      {err && (
        <pre style={{ color: "rgba(244,63,94,0.95)", whiteSpace: "pre-wrap", marginTop: 10 }}>
{err}
        </pre>
      )}
    </div>
  );
}
