from __future__ import annotations

import argparse
import csv
import json
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import SessionLocal
from app.models import Edge, Market


def _chunks(lst: list[dict[str, Any]], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def ensure_markets(db, market_ids: set[str], *, chunk_size: int = 5000) -> None:
    if not market_ids:
        return

    rows: list[dict[str, Any]] = [
        {
            "market_id": mid,
            "question": f"Market {mid}",
            "active": True,
            "closed": False,
        }
        for mid in market_ids
    ]

    for chunk in _chunks(rows, chunk_size):
        stmt = (
            pg_insert(Market)
            .values(chunk)
            .on_conflict_do_nothing(index_elements=["market_id"])
        )
        db.execute(stmt)
        db.commit()


def import_edges(db, edges: list[dict[str, Any]]) -> int:
    if not edges:
        return 0

    base = pg_insert(Edge)
    stmt = base.values(edges).on_conflict_do_update(
        constraint="uix_edge",
        set_={"weight": base.excluded.weight, "meta": base.excluded.meta},
    )

    db.execute(stmt)
    db.commit()
    return len(edges)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a final edge-list CSV (src_market_id,dst_market_id,edge_type,weight,meta) into Postgres."
    )
    parser.add_argument(
        "--csv",
        default="edges_final.csv",
        help="Path to CSV (default: edges_final.csv).",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=2000,
        help="Rows per insert batch (default: 2000).",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        total = 0
        batch: list[dict[str, Any]] = []
        pending_market_ids: set[str] = set()

        with open(args.csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"src_market_id", "dst_market_id", "edge_type", "weight", "meta"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise SystemExit(f"CSV missing required columns: {sorted(required)}")

            for row in reader:
                src = str(row["src_market_id"]).strip()
                dst = str(row["dst_market_id"]).strip()
                et = str(row["edge_type"]).strip() or "final"
                try:
                    w = float(row["weight"])
                except Exception:
                    continue

                meta_raw = row.get("meta")
                meta: Any
                if meta_raw is None or meta_raw == "":
                    meta = None
                else:
                    try:
                        meta = json.loads(meta_raw)
                    except Exception:
                        # Keep as string if not valid JSON
                        meta = {"raw": str(meta_raw)}

                pending_market_ids.add(src)
                pending_market_ids.add(dst)

                batch.append(
                    {
                        "src_market_id": src,
                        "dst_market_id": dst,
                        "edge_type": et,
                        "weight": w,
                        "meta": meta,
                    }
                )

                if len(batch) >= args.chunk:
                    ensure_markets(db, pending_market_ids)
                    pending_market_ids.clear()
                    total += import_edges(db, batch)
                    batch = []

        if batch:
            ensure_markets(db, pending_market_ids)
            total += import_edges(db, batch)

        print(f"Imported edges: {total}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
