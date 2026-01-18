from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

EPS = 1e-9


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def logit(p: float) -> float:
    p2 = float(p)
    if p2 < 1e-6:
        p2 = 1e-6
    if p2 > 1.0 - 1e-6:
        p2 = 1.0 - 1e-6
    return math.log(p2 / (1.0 - p2))


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def median_and_mad(values: np.ndarray) -> Tuple[float, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return 0.0, 0.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return med, mad


def rolling_median_mad(series: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    def _med(x: np.ndarray) -> float:
        m, _ = median_and_mad(x)
        return m

    def _mad(x: np.ndarray) -> float:
        _, d = median_and_mad(x)
        return d

    minp = max(5, window // 3)
    med = series.rolling(window=window, min_periods=minp).apply(_med, raw=True)
    mad = series.rolling(window=window, min_periods=minp).apply(_mad, raw=True)
    return med, mad


def robust_z(x: float, med: float, mad: float) -> float:
    denom = 1.4826 * mad + EPS
    return (x - med) / denom


def tokenize_title(title: str) -> List[str]:
    if title is None:
        return []
    t = str(title).lower()
    out: List[str] = []
    cur: List[str] = []
    for ch in t:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return [w for w in out if len(w) >= 3]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa.intersection(sb))
    union = len(sa.union(sb))
    if union == 0:
        return 0.0
    return inter / union


@dataclass(frozen=True)
class AlertPolicy:
    score_threshold: float = 0.90
    persistence_k: int = 2
    persistence_m: int = 3

    corr_confirm_threshold: float = 0.85
    price_confirm_threshold: float = 0.80
    liq_confirm_threshold: float = 0.60

    ml_confirm_threshold: float = 0.75
    deep_confirm_threshold: float = 0.75
    hybrid_confirm_threshold: float = 0.75


def persistence_trigger(scores: List[float], tau: float, k: int, m: int) -> bool:
    if len(scores) < m:
        return False
    tail = scores[-m:]
    cnt = 0
    for v in tail:
        if v >= tau:
            cnt += 1
    return cnt >= k
