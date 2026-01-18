import React from "react";
import { portfolioEVandVarGrouped } from "../stats";

export default function PortfolioStats({ positions }) {
  const { ev, variance, relativeVariance } =
    portfolioEVandVarGrouped(positions);

  const stddev = Math.sqrt(variance);

  return (
    <div className="row row-cols-2 row-cols-md-4 g-3">
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
    <div className="col">
      <div className="border rounded-3 p-3 bg-white">
        <div className="text-muted small">{label}</div>
        <div className="fs-5 fw-bold">{value}</div>
      </div>
    </div>
  );
}
