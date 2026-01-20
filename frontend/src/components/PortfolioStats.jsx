import React from "react";
import { portfolioEVandVarGrouped } from "../stats";

export default function PortfolioStats({ positions }) {
  const { ev, variance, relativeVariance } =
    portfolioEVandVarGrouped(positions);

  const stddev = Math.sqrt(variance);

  return (
    <div className="kpiGrid">
      <KPI label="Expected payout" value={`$${ev.toFixed(2)}`} />
      <KPI label="Variance" value={variance.toFixed(2)} />
      <KPI label="Std dev" value={stddev.toFixed(2)} />
      <KPI
        label="Var / E²"
        value={relativeVariance === null ? "—" : relativeVariance.toFixed(4)}
      />
    </div>
  );
}

function KPI({ label, value }) {
  return (
    <div className="card" style={{ padding: 14 }}>
      <div className="kpiLabel">{label}</div>
      <div className="kpiValue">{value}</div>
    </div>
  );
}
