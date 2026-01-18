from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from graph_builder import GraphConfig, build_market_graph
from polymarket_anomaly import DetectorConfig, compute_market_scores, generate_alerts
from utils import AlertPolicy

from fetch_polymarket import FetchConfig, fetch_and_write


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--meta_csv", default="data/polymarket_metadata.csv")
    p.add_argument("--ts_csv", default="data/polymarket_timeseries.csv")

    p.add_argument("--fetch", action="store_true")
    p.add_argument("--interval", default="1w")
    p.add_argument("--fidelity", type=int, default=5)
    p.add_argument("--start_ts", type=int, default=0)
    p.add_argument("--end_ts", type=int, default=0)
    p.add_argument("--no_book", action="store_true")

    # NEW: control Gamma paging/filters
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--max_pages", type=int, default=5)
    p.add_argument("--include_closed", action="store_true")

    p.add_argument("--out_scores", default="scores.csv")
    p.add_argument("--out_alerts", default="alerts.csv")
    p.add_argument("--out_alerts_json", default="alerts.json")

    p.add_argument("--rolling_window", type=int, default=288)

    p.add_argument("--score_threshold", type=float, default=0.90)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--m", type=int, default=3)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.fetch:
        os.makedirs(os.path.dirname(args.meta_csv) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(args.ts_csv) or ".", exist_ok=True)

        start_ts = None if args.start_ts == 0 else args.start_ts
        end_ts = None if args.end_ts == 0 else args.end_ts

        fcfg = FetchConfig(
            interval=args.interval,
            fidelity_minutes=args.fidelity,
            start_ts=None if args.start_ts == 0 else args.start_ts,
            end_ts=None if args.end_ts == 0 else args.end_ts,
            fetch_book=(not args.no_book),
            limit=args.limit,
            max_pages=args.max_pages,
            include_closed=args.include_closed,
        )

        fetch_and_write(args.meta_csv, args.ts_csv, fcfg)

        if os.path.getsize(args.meta_csv) == 0:
            raise RuntimeError(f"Fetch produced empty file: {args.meta_csv}")
        if os.path.getsize(args.ts_csv) == 0:
            raise RuntimeError(f"Fetch produced empty file: {args.ts_csv}")

    meta = pd.read_csv(args.meta_csv)
    ts = pd.read_csv(args.ts_csv)

    graph = build_market_graph(meta, GraphConfig())

    det_cfg = DetectorConfig(rolling_window=args.rolling_window, device="cpu")
    scored = compute_market_scores(ts, graph, det_cfg)

    policy = AlertPolicy(score_threshold=args.score_threshold, persistence_k=args.k, persistence_m=args.m)
    alerts = generate_alerts(scored, policy)

    scored.to_csv(args.out_scores, index=False)
    alerts.to_csv(args.out_alerts, index=False)

    records = []
    for _, r in alerts.iterrows():
        records.append(
            {
                "timestamp": str(r["timestamp"]),
                "market_id": str(r["market_id"]),
                "p": float(r["p"]),
                "anomaly_score": float(r["anomaly_score"]),
                "explanation": str(r["explanation"]),
            }
        )

    with open(args.out_alerts_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {args.out_scores}")
    print(f"Wrote {args.out_alerts}")
    print(f"Wrote {args.out_alerts_json}")
    print(f"Alerts found: {len(alerts)}")


if __name__ == "__main__":
    main()

