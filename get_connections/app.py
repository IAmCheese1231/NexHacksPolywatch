from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Query


CSV_PATH = os.getenv("EDGES_CSV_PATH", "edges.csv")  # set env var or put file as edges.csv


@dataclass(frozen=True)
class Edge:
    dst_market_id: int
    weight: float

    @property
    def correlation(self) -> float:
        # correlation = 1 - e^(-weight)
        return 1.0 - math.exp(-self.weight)


def load_graph(csv_path: str) -> Dict[int, List[Edge]]:
    """
    Returns adjacency list: src_market_id -> list of outgoing edges
    """
    graph: Dict[int, List[Edge]] = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"src_market_id", "dst_market_id", "weight"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(
                f"CSV header must be exactly: {','.join(sorted(expected))}. "
                f"Got: {reader.fieldnames}"
            )

        for row in reader:
            try:
                src = int(row["src_market_id"])
                dst = int(row["dst_market_id"])
                w = float(row["weight"])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Bad row: {row}") from e

            graph.setdefault(src, []).append(Edge(dst_market_id=dst, weight=w))

    # Optional: sort each adjacency list once at startup for faster queries
    for src, edges in graph.items():
        edges.sort(key=lambda e: e.weight, reverse=True)

    return graph


app = FastAPI(title="Market Neighbor API", version="1.0.0")
GRAPH: Dict[int, List[Edge]] = {}


@app.on_event("startup")
def startup():
    global GRAPH
    if not os.path.exists(CSV_PATH):
        raise RuntimeError(
            f"CSV file not found at {CSV_PATH}. "
            f"Set EDGES_CSV_PATH or place file at {os.path.abspath(CSV_PATH)}"
        )
    GRAPH = load_graph(CSV_PATH)


@app.get("/neighbors/{market_id}")
def get_neighbors(
    market_id: int,
    limit: int = Query(25, ge=1, le=1000),
):
    if market_id not in GRAPH:
        raise HTTPException(status_code=404, detail=f"market_id {market_id} not found")

    edges = GRAPH[market_id][:limit]
    return {
        "market_id": market_id,
        "limit": limit,
        "results": [
            {
                "dst_market_id": e.dst_market_id,
                "weight": e.weight,
                "correlation": e.correlation,
            }
            for e in edges
        ],
    }
