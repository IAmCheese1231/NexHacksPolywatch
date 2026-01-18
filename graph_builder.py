from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from utils import tokenize_title, jaccard


@dataclass
class GraphConfig:
    """
    Controls how market-to-market correlation edges are formed.
    """
    title_sim_threshold: float = 0.25
    title_edge_weight: float = 0.30
    group_edge_weight: float = 0.70
    max_neighbors: int = 15


def build_market_graph(
    meta: pd.DataFrame,
    cfg: GraphConfig,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Returns adjacency list:
      graph[market_id] = [(neighbor_market_id, normalized_weight), ...]
    """

    required = {"market_id", "title", "group_id"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")

    df = meta.copy()
    df["market_id"] = df["market_id"].astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["group_id"] = df["group_id"].fillna("").astype(str)

    market_ids = df["market_id"].tolist()

    # tokenize titles once
    title_tokens = {
        mid: tokenize_title(title)
        for mid, title in zip(df["market_id"], df["title"])
    }

    # group-based adjacency
    group_map: Dict[str, List[str]] = {}
    for mid, gid in zip(df["market_id"], df["group_id"]):
        if gid:
            group_map.setdefault(gid, []).append(mid)

    graph: Dict[str, List[Tuple[str, float]]] = {mid: [] for mid in market_ids}

    # --- group edges ---
    for gid, members in group_map.items():
        n = len(members)
        if n <= 1:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                a = members[i]
                b = members[j]
                graph[a].append((b, cfg.group_edge_weight))
                graph[b].append((a, cfg.group_edge_weight))

    # --- title similarity edges (O(n^2), MVP-safe) ---
    n = len(market_ids)
    for i in range(n):
        a = market_ids[i]
        ta = title_tokens[a]
        if not ta:
            continue
        for j in range(i + 1, n):
            b = market_ids[j]
            tb = title_tokens[b]
            if not tb:
                continue
            sim = jaccard(ta, tb)
            if sim >= cfg.title_sim_threshold:
                w = cfg.title_edge_weight * sim
                graph[a].append((b, w))
                graph[b].append((a, w))

    # --- cap neighbors + normalize ---
    for mid in graph:
        neighbors = graph[mid]
        if not neighbors:
            continue
        neighbors.sort(key=lambda x: x[1], reverse=True)
        neighbors = neighbors[: cfg.max_neighbors]

        total = sum(w for _, w in neighbors)
        if total > 0:
            neighbors = [(nid, w / total) for nid, w in neighbors]

        graph[mid] = neighbors

    return graph
