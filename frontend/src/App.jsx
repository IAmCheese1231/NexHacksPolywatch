import React, { useEffect, useState } from "react";
import AddPosition from "./components/AddPosition";
import PortfolioTable from "./components/PortfolioTable";
import PortfolioStats from "./components/PortfolioStats";
import PortfolioHistogram from "./components/PortfolioHistogram";
import { clearPortfolio, getPortfolio } from "./api";

export default function App() {
  const [positions, setPositions] = useState([]);
  const [err, setErr] = useState("");

  async function refresh() {
    setErr("");
    try {
      const p = await getPortfolio();
      setPositions(p);
    } catch (e) {
      setErr(String(e?.message || e));
    }
  }

  async function handleClear() {
    await clearPortfolio();
    await refresh();
  }

  useEffect(() => {
    refresh();
  }, []);

  return (
    <div className="bg-light min-vh-100">
      <div className="container py-4">

        {/* ===== Header ===== */}
        <div className="d-flex flex-column flex-md-row justify-content-between align-items-md-end gap-3 mb-4">
          <div>
            <div className="text-uppercase text-muted small">
              Polymarket Portfolio • Risk
            </div>
            <h1 className="h3 mb-1">Portfolio Risk Dashboard</h1>
            <div className="text-muted">
              Terminal payout distribution using implied probabilities
            </div>
          </div>

          <div className="d-flex gap-2">
            <button
              className="btn btn-outline-secondary"
              onClick={refresh}
            >
              Refresh
            </button>
            <button
              className="btn btn-danger"
              onClick={handleClear}
            >
              Clear portfolio
            </button>
          </div>
        </div>

        {/* ===== Add Position ===== */}
        <div className="row mb-3">
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-body">
                <h2 className="h6 mb-3">Add position</h2>
                <AddPosition onAdded={refresh} />
              </div>
            </div>
          </div>
        </div>

        {/* ===== Stats ===== */}
        <div className="row mb-3">
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-body">
                <h2 className="h6 mb-3">Portfolio stats</h2>
                <PortfolioStats positions={positions} />
              </div>
            </div>
          </div>
        </div>

        {/* ===== Histogram (wide) ===== */}
        <div className="row mb-3">
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-body">
                <div className="d-flex justify-content-between align-items-baseline mb-2">
                  <h2 className="h6 mb-0">Monte Carlo histogram</h2>
                  <span className="text-muted small">
                    0 → 5× EV (plus overflow)
                  </span>
                </div>
                <PortfolioHistogram positions={positions} />
              </div>
            </div>
          </div>
        </div>

        {/* ===== Positions Table ===== */}
        <div className="row">
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-body">
                <h2 className="h6 mb-3">Positions</h2>
                <PortfolioTable positions={positions} />
              </div>
            </div>
          </div>
        </div>

        {/* ===== Error ===== */}
        {err && (
          <div className="alert alert-danger mt-4 mb-0">
            <div className="fw-semibold">Error</div>
            <pre className="mb-0">{err}</pre>
          </div>
        )}

        {/* ===== Footer note ===== */}
        <div className="text-muted small mt-4">
          Assumes independence across markets; outcomes within each market are
          mutually exclusive.
        </div>

      </div>
    </div>
  );
}
