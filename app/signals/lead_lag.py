# /srv/app/app/signals/lead_lag.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
from sqlalchemy import text

# -----------------------------
# Configuration knobs
# -----------------------------
DEFAULT_LOOKBACK_POINTS = 240     # e.g. last 240 samples as baseline
DEFAULT_RECENT_POINTS = 12        # e.g. last 12 samples as "recent"
DEFAULT_MIN_POINTS = 80

# If you want a bounded "correlation factor" in [0,1] from your graph weight:
def corr_factor_from_weight(w: float) -> float:
    # monotonic squashing; change slope if you want more/less aggressive mapping
    # w=0.8 -> ~0.55, w=1.2 -> ~0.70, w=2 -> ~0.86
    return float(1.0 - math.exp(-max(w, 0.0)))

@dataclass
class MoveStats:
    market_id: str
    last_price: float
    recent_ret: float          # sum of log returns over recent window
    z: float                   # zscore of recent_ret vs baseline
    n: int                     # points used

@dataclass
class NeighborIdea:
    src_market_id: str
    dst_market_id: str
    edge_weight: float
    corr_factor: float
    src_z: float
    dst_z: float
    lag_score: float           # higher = more "hasn't reacted"
    direction: str             # "buy_dst" or "sell_dst" (based on src move)
    explain: Dict[str, Any]

# -----------------------------
# DB fetch helpers
# -----------------------------
def _fetch_recent_prices(db, market_id: str, limit: int) -> List[Tuple[int, float]]:
    # You MUST adapt this query to your schema.
    # Expected: ts increasing, price numeric.
    q = text("""
        SELECT ts, price
        FROM price_points
        WHERE market_id = :mid
        ORDER BY ts DESC
        LIMIT :lim
    """)
    rows = db.execute(q, {"mid": market_id, "lim": limit}).fetchall()
    # rows come newest-first; reverse
    rows = list(reversed([(int(r[0]), float(r[1])) for r in rows]))
    return rows

def _compute_move_stats_from_prices(
    market_id: str,
    pts: List[Tuple[int, float]],
    lookback_points: int,
    recent_points: int,
    min_points: int,
) -> Optional[MoveStats]:
    if len(pts) < max(min_points, recent_points + 2):
        return None

    px = np.array([p for _, p in pts], dtype=np.float64)
    px = np.clip(px, 1e-12, None)
    logret = np.diff(np.log(px))  # length = n-1

    # baseline window: exclude most recent 'recent_points' returns
    if len(logret) <= recent_points + 5:
        return None

    recent = logret[-recent_points:].sum()
    base = logret[-(lookback_points + recent_points):-recent_points] if len(logret) > lookback_points + recent_points else logret[:-recent_points]

    mu = float(base.mean())
    sd = float(base.std(ddof=1)) if base.size >= 2 else 0.0
    if sd < 1e-10:
        z = 0.0
    else:
        z = float((recent - mu) / sd)

    return MoveStats(
        market_id=str(market_id),
        last_price=float(px[-1]),
        recent_ret=float(recent),
        z=z,
        n=len(pts),
    )

def get_move_stats(
    db,
    market_id: str,
    lookback_points: int = DEFAULT_LOOKBACK_POINTS,
    recent_points: int = DEFAULT_RECENT_POINTS,
    min_points: int = DEFAULT_MIN_POINTS,
) -> Optional[MoveStats]:
    pts = _fetch_recent_prices(db, market_id, limit=lookback_points + recent_points + 5)
    return _compute_move_stats_from_prices(market_id, pts, lookback_points, recent_points, min_points)

def fetch_neighbors(
    db,
    market_id: str,
    limit: int = 50,
    min_weight: float = 0.15,
) -> List[Tuple[str, float]]:
    q = text("""
        SELECT dst_market_id, weight
        FROM edges
        WHERE edge_type='final' AND src_market_id = :mid AND weight >= :mw
        ORDER BY weight DESC
        LIMIT :lim
    """)
    rows = db.execute(q, {"mid": str(market_id), "mw": float(min_weight), "lim": int(limit)}).fetchall()
    return [(str(r[0]), float(r[1])) for r in rows]

# -----------------------------
# Core lead/lag idea engine
# -----------------------------
def lead_lag_ideas_for_move(
    db,
    src_market_id: str,
    neighbor_limit: int = 50,
    min_edge_weight: float = 0.15,
    min_src_z: float = 2.0,          # “abnormal”
    max_abs_dst_z: float = 1.0,      # “hasn’t moved much”
    lookback_points: int = DEFAULT_LOOKBACK_POINTS,
    recent_points: int = DEFAULT_RECENT_POINTS,
) -> Dict[str, Any]:
    src = get_move_stats(db, src_market_id, lookback_points=lookback_points, recent_points=recent_points)
    if not src:
        return {"ok": False, "reason": "not_enough_price_points_for_src"}

    if abs(src.z) < min_src_z:
        return {"ok": False, "reason": "src_move_not_abnormal", "src": src.__dict__}

    nbrs = fetch_neighbors(db, src_market_id, limit=neighbor_limit, min_weight=min_edge_weight)
    if not nbrs:
        return {"ok": False, "reason": "no_neighbors_found", "src": src.__dict__}

    ideas: List[NeighborIdea] = []

    for dst_id, w in nbrs:
        dst = get_move_stats(db, dst_id, lookback_points=lookback_points, recent_points=recent_points)
        if not dst:
            continue

        # we want "laggards": src moved a lot, dst didn't
        if abs(dst.z) > max_abs_dst_z:
            continue

        corr = corr_factor_from_weight(w)

        # lag_score: combine how abnormal src is, how quiet dst is, and the affinity
        lag_score = corr * (abs(src.z) - abs(dst.z))

        # direction heuristic:
        # - if src z positive (price up), suggest "buy dst" (if you believe info should propagate)
        # - if src z negative, suggest "sell dst"
        direction = "buy_dst" if src.z > 0 else "sell_dst"

        ideas.append(
            NeighborIdea(
                src_market_id=str(src_market_id),
                dst_market_id=str(dst_id),
                edge_weight=float(w),
                corr_factor=float(corr),
                src_z=float(src.z),
                dst_z=float(dst.z),
                lag_score=float(lag_score),
                direction=direction,
                explain={
                    "why": "src moved abnormally; dst is strongly linked but hasn't reacted",
                    "src_recent_ret": src.recent_ret,
                    "dst_recent_ret": dst.recent_ret,
                    "mapping": "corr_factor = 1 - exp(-edge_weight)",
                },
            )
        )

    ideas.sort(key=lambda x: x.lag_score, reverse=True)

    return {
        "ok": True,
        "src": src.__dict__,
        "params": {
            "neighbor_limit": neighbor_limit,
            "min_edge_weight": min_edge_weight,
            "min_src_z": min_src_z,
            "max_abs_dst_z": max_abs_dst_z,
            "lookback_points": lookback_points,
            "recent_points": recent_points,
        },
        "ideas": [i.__dict__ for i in ideas[:25]],
    }
