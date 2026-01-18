from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


@dataclass(frozen=True)
class FetchConfig:
    limit: int = 100
    max_pages: int = 5
    include_closed: bool = False

    fidelity_minutes: int = 5
    interval: str = "1w"
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None

    fetch_book: bool = True
    depth_levels: int = 5

    sleep_s: float = 0.2
    timeout_s: float = 20.0


def gamma_health(cfg: FetchConfig) -> bool:
    r = requests.get(f"{GAMMA_BASE}/status", timeout=cfg.timeout_s)
    return r.status_code == 200 and "OK" in r.text


def fetch_markets(cfg: FetchConfig) -> List[dict]:
    markets: List[dict] = []
    offset = 0

    for _ in range(cfg.max_pages):
        params: Dict[str, Any] = {"limit": cfg.limit, "offset": offset}
        if not cfg.include_closed:
            params["closed"] = "false"

        r = requests.get(f"{GAMMA_BASE}/markets", params=params, timeout=cfg.timeout_s)
        r.raise_for_status()
        page = r.json()

        if not isinstance(page, list) or not page:
            break

        markets.extend(page)
        offset += len(page)
        time.sleep(cfg.sleep_s)

    return markets


def _parse_clob_token_ids(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if isinstance(raw, str):
        try:
            j = json.loads(raw)
            if isinstance(j, list):
                return [str(x) for x in j if str(x).strip()]
        except Exception:
            pass
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []


def extract_metadata(markets: List[dict]) -> pd.DataFrame:
    rows: List[dict] = []

    for m in markets:
        if m.get("enableOrderBook") is not True:
            continue

        vol = float(m.get("volumeNum", 0) or 0)
        liq = float(m.get("liquidityNum", 0) or 0)
        if vol < 1000 or liq < 500:
            continue

        mid = str(m.get("id", "")).strip()
        if not mid:
            continue

        tokens = _parse_clob_token_ids(m.get("clobTokenIds"))
        if not tokens:
            continue

        rows.append(
            {
                "market_id": mid,
                "title": m.get("question") or m.get("title") or "",
                "group_id": str(m.get("conditionId") or ""),
                "yes_token_id": tokens[0],
                "no_token_id": tokens[1] if len(tokens) > 1 else "",
            }
        )

    return pd.DataFrame(rows)


def fetch_prices_history(token_id: str, cfg: FetchConfig) -> List[Tuple[int, float]]:
    def _call(interval: str, fidelity: int) -> List[Tuple[int, float]]:
        params = {"market": token_id, "fidelity": fidelity, "interval": interval}
        r = requests.get(f"{CLOB_BASE}/prices-history", params=params, timeout=cfg.timeout_s)
        r.raise_for_status()
        data = r.json()
        out = []
        for pt in data.get("history", []):
            try:
                out.append((int(pt["t"]), float(pt["p"])))
            except Exception:
                pass
        return out

    out = _call(cfg.interval, cfg.fidelity_minutes)
    if out:
        return out

    # fallback: longer range + coarser granularity
    return _call("max", max(60, cfg.fidelity_minutes))


def fetch_order_book(token_id: str, cfg: FetchConfig) -> Dict[str, Any]:
    r = requests.get(
        f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=cfg.timeout_s
    )
    r.raise_for_status()
    return r.json()


def _best_price_and_depth(side: list, k: int) -> Tuple[Optional[float], float]:
    best, depth = None, 0.0
    for i, lvl in enumerate(side):
        try:
            p, s = float(lvl["price"]), float(lvl["size"])
        except Exception:
            continue
        if best is None:
            best = p
        if i < k:
            depth += s
    return best, depth


def build_timeseries(meta: pd.DataFrame, cfg: FetchConfig) -> pd.DataFrame:
    rows = []

    for _, r in meta.iterrows():
        mid = r["market_id"]
        token = r["yes_token_id"]

        hist = fetch_prices_history(token, cfg)
        if not hist:
            print(f"[fetch] empty history for token {token}")
            continue

        bid = ask = spread = None
        depth_bid = depth_ask = 0.0

        if cfg.fetch_book:
            book = fetch_order_book(token, cfg)
            bid, depth_bid = _best_price_and_depth(book.get("bids", []), cfg.depth_levels)
            ask, depth_ask = _best_price_and_depth(book.get("asks", []), cfg.depth_levels)
            if bid and ask:
                spread = ask - bid

        for t, p in hist:
            rows.append(
                {
                    "timestamp": pd.to_datetime(t, unit="s", utc=True).isoformat(),
                    "market_id": mid,
                    "p": p,
                    "bid": bid,
                    "ask": ask,
                    "spread": spread,
                    "depth_bid": depth_bid,
                    "depth_ask": depth_ask,
                }
            )

        time.sleep(cfg.sleep_s)

    return pd.DataFrame(rows)


def fetch_and_write(meta_csv: str, ts_csv: str, cfg: FetchConfig):
    if not gamma_health(cfg):
        raise RuntimeError("Gamma health check failed")

    markets = fetch_markets(cfg)
    print(f"[fetch] gamma markets fetched: {len(markets)}")

    meta = extract_metadata(markets)
    print(f"[fetch] markets with usable clobTokenIds: {len(meta)}")

    if meta.empty:
        raise RuntimeError("No usable markets after filtering")

    ts = build_timeseries(meta, cfg)
    print(f"[fetch] timeseries rows: {len(ts)}")

    if ts.empty:
        raise RuntimeError("No historical price data returned")

    os.makedirs(os.path.dirname(meta_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(ts_csv) or ".", exist_ok=True)

    meta.to_csv(meta_csv, index=False)
    ts.to_csv(ts_csv, index=False)

    return meta, ts
