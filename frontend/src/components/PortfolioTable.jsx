import React from "react";
import { prettyProb } from "../utils";

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
  return (
    <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 16 }}>
      <h2 style={{ marginTop: 0 }}>Portfolio</h2>

      {positions.length === 0 ? (
        <div style={{ opacity: 0.7 }}>No positions yet.</div>
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ textAlign: "left" }}>
              <th style={{ borderBottom: "1px solid #eee", padding: 8 }}>Market</th>
              <th style={{ borderBottom: "1px solid #eee", padding: 8 }}>Outcome</th>
              <th style={{ borderBottom: "1px solid #eee", padding: 8 }}>Shares</th>
              <th style={{ borderBottom: "1px solid #eee", padding: 8 }}>Implied prob</th>
              <th style={{ borderBottom: "1px solid #eee", padding: 8 }}>E[payout]</th>
              <th style={{ borderBottom: "1px solid #eee", padding: 8 }}>Var(payout)</th>
            </tr>
          </thead>

          <tbody>
            {positions.map((p, idx) => {
              const ev = positionEV(p);
              const vr = positionVar(p);

              return (
                <tr key={idx}>
                  <td style={{ borderBottom: "1px solid #f3f3f3", padding: 8 }}>
                    {p.market_title}
                    <div style={{ fontSize: 12, opacity: 0.65 }}>
                      <code>{p.market_slug}</code>
                    </div>
                  </td>

                  <td style={{ borderBottom: "1px solid #f3f3f3", padding: 8 }}>{p.outcome_label}</td>

                  <td style={{ borderBottom: "1px solid #f3f3f3", padding: 8 }}>{p.shares}</td>

                  <td style={{ borderBottom: "1px solid #f3f3f3", padding: 8 }}>
                    {prettyProb(p.implied_probability)}
                  </td>

                  <td style={{ borderBottom: "1px solid #f3f3f3", padding: 8 }}>
                    {ev === null ? "—" : `$${ev.toFixed(2)}`}
                  </td>

                  <td style={{ borderBottom: "1px solid #f3f3f3", padding: 8 }}>
                    {vr === null ? "—" : vr.toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
