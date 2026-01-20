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
    const id = window.setTimeout(() => {
      refresh();
    }, 0);
    return () => window.clearTimeout(id);
  }, []);

  useEffect(() => {
    const root = document.getElementById("root");
    if (!root) return;

    const postHeight = () => {
      // Measure *content* height only (avoid using offsetHeight which includes viewport).
      const height = root.scrollHeight;
      if (height > 0) {
        window.parent?.postMessage(
          { type: "POLYWATCH_PORTFOLIO_HEIGHT", height },
          "*"
        );
      }
    };

    // Initial + next tick (after fonts/layout settle)
    postHeight();
    const raf = window.requestAnimationFrame(postHeight);

    let resizeObserver;
    if ("ResizeObserver" in window) {
      resizeObserver = new ResizeObserver(() => postHeight());
      resizeObserver.observe(root);
    }

    window.addEventListener("resize", postHeight);
    return () => {
      window.cancelAnimationFrame(raf);
      window.removeEventListener("resize", postHeight);
      resizeObserver?.disconnect?.();
    };
  }, []);

  return (
    <div>
      <div className="container">
        <div className="topbar">
          <div>
            <div className="h1">Dashboard</div>
          </div>

          <div className="row" style={{ justifyContent: "flex-end" }}>
            <button className="button" onClick={refresh}>
              Refresh
            </button>
            <button className="button" onClick={handleClear}>
              Clear portfolio
            </button>
          </div>
        </div>

        <div className="card">
          <h2>Add position</h2>
          <AddPosition onAdded={refresh} />
        </div>

        <div style={{ height: 12 }} />

        <div className="card">
          <h2>Portfolio stats</h2>
          <PortfolioStats positions={positions} />
        </div>

        <div style={{ height: 12 }} />

        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2 style={{ margin: 0 }}>Monte Carlo histogram</h2>
          </div>
          <div style={{ height: 10 }} />
          <PortfolioHistogram positions={positions} />
        </div>

        <div style={{ height: 12 }} />

        <div className="card">
          <h2>Positions</h2>
          <PortfolioTable positions={positions} />
        </div>

        {err && (
          <div className="card" style={{ borderColor: "rgba(244,63,94,0.25)" }}>
            <h2>Error</h2>
            <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{err}</pre>
          </div>
        )}

        <div className="subtle" style={{ marginTop: 14 }}>
          Assumes independence across markets; outcomes within each market are
          mutually exclusive.
        </div>
      </div>
    </div>
  );
}
