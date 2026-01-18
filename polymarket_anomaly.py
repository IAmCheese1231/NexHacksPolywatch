from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import (
    rolling_median_mad,
    robust_z,
    sigmoid,  # scalar sigmoid (we'll wrap it for arrays)
)

from ml_classical import ClassicalConfig, ClassicalDetectors
from deep_autoencoder import AEConfig, AutoencoderDetector
from deep_forecast import ForecastConfig, SequenceForecastDetector

EPS = 1e-9


# -------------------------
# Configs
# -------------------------

@dataclass
class DetectorConfig:
    # Rolling stats / baseline
    rolling_window: int = 288  # for 5-min cadence: 288 ~ 1 day
    # Correlation / graph-ish layer (lightweight)
    corr_lookback: int = 288
    corr_min_pairs: int = 120
    corr_threshold: float = 0.65
    corr_break_z: float = 3.0

    # CUSUM drift
    cusum_k: float = 0.5
    cusum_h: float = 6.0
    cusum_scale: float = 1.0

    # Weights inside "base" score
    w_price: float = 0.35
    w_liq: float = 0.25
    w_corr: float = 0.25
    w_cusum: float = 0.15

    # Layer weights for final anomaly score
    w_base_layer: float = 0.45
    w_ml_layer: float = 0.20
    w_deep_layer: float = 0.25
    w_hybrid_layer: float = 0.10

    # Deep model device
    device: str = "cpu"


@dataclass
class AlertPolicy:
    score_threshold: float = 0.85
    persistence_k: int = 3
    persistence_m: int = 5

    corr_confirm_threshold: float = 0.7
    price_confirm_threshold: float = 0.6
    liq_confirm_threshold: float = 0.6

    ml_confirm_threshold: float = 0.65
    deep_confirm_threshold: float = 0.65
    hybrid_confirm_threshold: float = 0.65


# -------------------------
# Small helpers
# -------------------------

def _safe_logit(p: float, eps: float = 1e-6) -> float:
    """
    Numerically stable logit for probabilities.
    """
    p = min(max(float(p), eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


def _sigmoid_np(x):
    """
    Vector-safe sigmoid.
    Uses numpy for arrays; falls back to scalar utils.sigmoid for scalars.
    """
    if np.isscalar(x):
        return float(sigmoid(float(x)))
    x = np.asarray(x, dtype=float)
    # stable logistic
    return 1.0 / (1.0 + np.exp(-x))


def _score_abs_z(z: float, z0: float = 3.0) -> float:
    """
    Map |z| to [0,1], where values past z0 rapidly approach 1.
    """
    return float(np.clip(_sigmoid_np(abs(z) - z0), 0.0, 1.0))


def _roll_med_mad(s: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    """
    Use repo's rolling_median_mad; ensure MAD is never 0/NaN.
    """
    med, mad = rolling_median_mad(s, window=window)
    mad = mad.replace(0.0, np.nan).ffill().fillna(1.0)
    return med, mad


def _compute_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "bid" in out.columns and "ask" in out.columns:
        out["spread"] = (out["ask"] - out["bid"]).astype(float).clip(lower=0.0)
    else:
        out["spread"] = np.nan

    if "depth_bid" in out.columns and "depth_ask" in out.columns:
        out["depth"] = (out["depth_bid"].astype(float) + out["depth_ask"].astype(float)).clip(lower=0.0)
    else:
        out["depth"] = np.nan

    return out


def persistence_trigger(recent: List[float], thr: float, k: int, m: int) -> bool:
    if len(recent) < m:
        return False
    window = recent[-m:]
    cnt = sum(1 for x in window if x >= thr)
    return cnt >= k


def _explain(row: pd.Series) -> str:
    parts = []
    parts.append(f"price={row.get('price_score', 0.0):.2f}")
    parts.append(f"liq={row.get('liq_score', 0.0):.2f}")
    parts.append(f"corr={row.get('corr_score', 0.0):.2f}")
    parts.append(f"cusum={row.get('cusum_score', 0.0):.2f}")
    parts.append(f"ml={row.get('ml_score', 0.0):.2f}")
    parts.append(f"deep={row.get('deep_score', 0.0):.2f}")
    parts.append(f"hybrid={row.get('hybrid_score', 0.0):.2f}")
    return " | ".join(parts)


# -------------------------
# Correlation-break layer
# -------------------------

def _corr_break_scores(df: pd.DataFrame, cfg: DetectorConfig) -> pd.DataFrame:
    """
    Approx correlation-break: build a static neighbor list using a trailing window
    correlation of returns, then compute residual r - weighted_avg(neighbors).

    Outputs:
      r_hat, r_resid, e_med, e_mad, z_e, corr_score
    """
    out = df.copy()
    out["r_hat"] = 0.0
    out["r_resid"] = 0.0

    wide = out.pivot(index="timestamp", columns="market_id", values="r").sort_index()

    if len(wide) < cfg.corr_min_pairs:
        out["e_med"] = 0.0
        out["e_mad"] = 1.0
        out["z_e"] = 0.0
        out["corr_score"] = 0.0
        return out

    tail = wide.iloc[-cfg.corr_lookback :] if len(wide) > cfg.corr_lookback else wide
    corr = tail.corr(min_periods=cfg.corr_min_pairs)

    neighbors: Dict[str, List[str]] = {}
    for m in corr.columns:
        c = corr[m].drop(index=m, errors="ignore")
        nb = c[c.abs() >= cfg.corr_threshold].sort_values(key=lambda x: x.abs(), ascending=False).index.tolist()
        neighbors[str(m)] = [str(x) for x in nb[:5]]

    # compute per-market r_hat and residual
    for mid, g in out.groupby("market_id", sort=False):
        mid_s = str(mid)
        nb = neighbors.get(mid_s, [])
        if not nb:
            r = g["r"].to_numpy(dtype=float)
            out.loc[g.index, "r_hat"] = 0.0
            out.loc[g.index, "r_resid"] = np.nan_to_num(r, nan=0.0)
            continue

        ws = []
        for j in nb:
            if (mid in corr.columns) and (j in corr.columns):
                ws.append(float(abs(corr.loc[mid, j])))
            else:
                ws.append(0.0)
        wsum = sum(ws) + EPS
        ws = [w / wsum for w in ws]

        rh = np.zeros(len(g), dtype=float)
        ts = g["timestamp"]
        for w, j in zip(ws, nb):
            if j in wide.columns:
                rj = wide[j].reindex(ts).to_numpy(dtype=float)
                rh += w * np.nan_to_num(rj, nan=0.0)

        r = g["r"].to_numpy(dtype=float)
        resid = np.nan_to_num(r, nan=0.0) - rh

        out.loc[g.index, "r_hat"] = rh
        out.loc[g.index, "r_resid"] = resid

    # robust z on residual (global rolling; simple but effective)
    em, ea = _roll_med_mad(out["r_resid"], cfg.rolling_window)
    out["e_med"] = em
    out["e_mad"] = ea
    out["z_e"] = (out["r_resid"] - out["e_med"]) / (1.4826 * out["e_mad"] + EPS)
    out["corr_score"] = out["z_e"].apply(lambda z: _score_abs_z(0.0 if pd.isna(z) else float(z), z0=cfg.corr_break_z))
    return out


# -------------------------
# Main scoring pipeline
# -------------------------

def compute_market_scores(
    df: pd.DataFrame,
    graph=None,
    cfg: DetectorConfig | None = None,
) -> pd.DataFrame:
    """
    Polymarket anomaly scoring.

    main.py passes (df, graph, cfg). We accept that signature; graph may be unused.

    Required columns:
      timestamp, market_id, p, bid, ask, depth_bid, depth_ask

    Produces:
      base_score, ml_score, deep_score, hybrid_score, anomaly_score,
      plus many intermediate features.
    """
    # Backward/forward compatibility:
    # if caller passed (df, cfg) and graph is actually DetectorConfig
    if cfg is None and isinstance(graph, DetectorConfig):
        cfg = graph
        graph = None
    if cfg is None:
        raise ValueError("DetectorConfig must be provided")

    out = df.copy()
    out = out.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    # stable p/logit
    out["p"] = out["p"].astype(float).clip(0.0001, 0.9999)
    out["logit_p"] = out["p"].apply(lambda x: _safe_logit(float(x)))

    # liquidity basics
    out = _compute_liquidity_features(out)

    # returns + spread delta per market
    out["r"] = 0.0
    out["d_spread"] = 0.0
    # placeholders for endpoints that don't provide these (kept but may be NaN)
    out["d_vol"] = np.nan
    out["d_trades"] = np.nan

    for mid, g in out.groupby("market_id", sort=False):
        lp = g["logit_p"].to_numpy(dtype=float)
        r = np.diff(lp, prepend=np.nan)
        out.loc[g.index, "r"] = r

        sp = g["spread"].to_numpy(dtype=float) if "spread" in g.columns else np.full(len(g), np.nan)
        out.loc[g.index, "d_spread"] = np.diff(sp, prepend=np.nan)

    # rolling robust stats
    out["r_med"], out["r_mad"] = _roll_med_mad(out["r"], cfg.rolling_window)
    out["v_med"], out["v_mad"] = _roll_med_mad(out["d_vol"], cfg.rolling_window)
    out["n_med"], out["n_mad"] = _roll_med_mad(out["d_trades"], cfg.rolling_window)
    out["s_med"], out["s_mad"] = _roll_med_mad(out["d_spread"], cfg.rolling_window)

    # robust z's
    out["z_r"] = (out["r"] - out["r_med"]) / (1.4826 * out["r_mad"] + EPS)
    out["z_v"] = (out["d_vol"] - out["v_med"]) / (1.4826 * out["v_mad"] + EPS)
    out["z_n"] = (out["d_trades"] - out["n_med"]) / (1.4826 * out["n_mad"] + EPS)
    out["z_s"] = (out["d_spread"] - out["s_med"]) / (1.4826 * out["s_mad"] + EPS)

    # baseline "price"
    out["price_score"] = out["z_r"].apply(lambda z: _score_abs_z(0.0 if pd.isna(z) else float(z)))

    # baseline "liquidity"
    spread_score = out["z_s"].apply(lambda z: _score_abs_z(0.0 if pd.isna(z) else float(z)))

    # depth anomaly (vector-safe sigmoid!)
    if "depth" in out.columns:
        depth = out["depth"].to_numpy(dtype=float)
        med_depth = float(np.nanmedian(depth)) if np.isfinite(np.nanmedian(depth)) else 0.0
        depth_score = np.nan_to_num(_sigmoid_np((med_depth - depth) / (med_depth + EPS)), nan=0.0)
    else:
        depth_score = np.zeros(len(out), dtype=float)

    out["liq_score"] = (0.7 * spread_score + 0.3 * depth_score).clip(0.0, 1.0)

    # correlation-break
    out = _corr_break_scores(out, cfg)

    # CUSUM drift per market on robust z of returns
    out["cusum_score"] = 0.0
    for mid, g in out.groupby("market_id", sort=False):
        c = 0.0
        scores = []
        for r, med, mad in zip(g["r"], g["r_med"], g["r_mad"]):
            if pd.isna(r) or pd.isna(med) or pd.isna(mad) or float(mad) <= 0.0:
                scores.append(0.0)
                continue
            zr = robust_z(float(r), float(med), float(mad))
            c = max(0.0, c + (zr - cfg.cusum_k) * cfg.cusum_scale)
            scores.append(float(np.clip(_sigmoid_np((c - cfg.cusum_h) / 2.0), 0.0, 1.0)))
        out.loc[g.index, "cusum_score"] = scores

    out["base_score"] = (
        cfg.w_price * out["price_score"]
        + cfg.w_liq * out["liq_score"]
        + cfg.w_corr * out["corr_score"]
        + cfg.w_cusum * out["cusum_score"]
    ).clip(0.0, 1.0)

    # -------------------------
    # ML + Deep per market
    # -------------------------

    out["iforest_score_01"] = 0.0
    out["lof_score_01"] = 0.0
    out["ocsvm_score_01"] = 0.0
    out["ae_score_01"] = 0.0
    out["gru_forecast_score_01"] = 0.0
    out["trf_forecast_score_01"] = 0.0
    out["hybrid_if_on_ae_01"] = 0.0

    candidates = [c for c in ["r", "spread", "d_spread", "r_resid", "price_score", "liq_score", "corr_score"] if c in out.columns]

    def usable_cols(g: pd.DataFrame, cols: List[str], min_frac: float = 0.90) -> List[str]:
        n = len(g)
        if n == 0:
            return []
        res: List[str] = []
        for c in cols:
            s = g[c]
            if float(s.notna().sum()) / float(n) < min_frac:
                continue
            rng = float(s.max(skipna=True) - s.min(skipna=True))
            if not np.isfinite(rng) or rng < 1e-12:
                continue
            res.append(c)
        return res

    classical_cfg = ClassicalConfig(train_window=cfg.rolling_window * 15)
    ae_cfg = AEConfig(train_window=cfg.rolling_window * 20, denoise_std=0.05, device=cfg.device)

    # forecast models on logit_p only
    fcfg = ForecastConfig(train_window=cfg.rolling_window * 30, seq_len=32, epochs=8, device=cfg.device)

    for mid, g in out.groupby("market_id", sort=False):
        g = g.copy()

        feats = usable_cols(g, candidates, min_frac=0.85)
        if not feats:
            feats = usable_cols(g, ["r", "spread", "d_spread"], min_frac=0.85)

        # Classical
        if len(feats) >= 1:
            cd = ClassicalDetectors(classical_cfg)
            cd.fit(g, feats)
            cls = cd.score(g, feats)
            out.loc[g.index, ["iforest_score_01", "lof_score_01", "ocsvm_score_01"]] = cls[
                ["iforest_score_01", "lof_score_01", "ocsvm_score_01"]
            ].values

        # Autoencoder
        ae_latent = None
        if len(feats) >= 1:
            ae = AutoencoderDetector(ae_cfg)
            ae.fit(g, feats)
            ae_out = ae.score_and_latent(g, feats)
            out.loc[g.index, "ae_score_01"] = ae_out["ae_score_01"].values

            latent_cols = [c for c in ae_out.columns if c.startswith("z")]
            if latent_cols:
                ae_latent = ae_out[latent_cols].copy()

        # Hybrid: IF on latent
        if ae_latent is not None and len(ae_latent.columns) > 0:
            cd_lat = ClassicalDetectors(classical_cfg)
            cd_lat.fit(ae_latent, list(ae_latent.columns))
            lat_scores = cd_lat.score(ae_latent, list(ae_latent.columns))
            out.loc[g.index, "hybrid_if_on_ae_01"] = lat_scores["iforest_score_01"].values

        # GRU + Transformer forecasts on logit_p
        if "logit_p" in g.columns and g["logit_p"].notna().mean() > 0.99:
            seq_cols = ["logit_p"]

            gru_det = SequenceForecastDetector(fcfg, model_type="gru")
            gru_det.fit(g, seq_cols)
            out.loc[g.index, "gru_forecast_score_01"] = gru_det.score(
                g, seq_cols, "gru_forecast_score_01"
            ).values

            trf_det = SequenceForecastDetector(fcfg, model_type="transformer")
            trf_det.fit(g, seq_cols)
            out.loc[g.index, "trf_forecast_score_01"] = trf_det.score(
                g, seq_cols, "trf_forecast_score_01"
            ).values

    out["ml_score"] = (
        0.40 * out["iforest_score_01"]
        + 0.30 * out["lof_score_01"]
        + 0.30 * out["ocsvm_score_01"]
    ).clip(0.0, 1.0)

    out["deep_score"] = (
        0.50 * out["ae_score_01"]
        + 0.25 * out["gru_forecast_score_01"]
        + 0.25 * out["trf_forecast_score_01"]
    ).clip(0.0, 1.0)

    out["hybrid_score"] = out["hybrid_if_on_ae_01"].clip(0.0, 1.0)

    out["anomaly_score"] = (
        cfg.w_base_layer * out["base_score"]
        + cfg.w_ml_layer * out["ml_score"]
        + cfg.w_deep_layer * out["deep_score"]
        + cfg.w_hybrid_layer * out["hybrid_score"]
    ).clip(0.0, 1.0)

    return out


def generate_alerts(scored: pd.DataFrame, policy: AlertPolicy) -> pd.DataFrame:
    df = scored.copy().sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    alerts = []
    for mid, g in df.groupby("market_id", sort=False):
        recent: List[float] = []
        for _, row in g.iterrows():
            s = float(row["anomaly_score"]) if pd.notna(row["anomaly_score"]) else 0.0
            recent.append(s)

            if not persistence_trigger(recent, policy.score_threshold, policy.persistence_k, policy.persistence_m):
                continue

            corr_ok = float(row.get("corr_score", 0.0)) >= policy.corr_confirm_threshold
            price_ok = float(row.get("price_score", 0.0)) >= policy.price_confirm_threshold
            liq_ok = float(row.get("liq_score", 0.0)) >= policy.liq_confirm_threshold

            ml_ok = float(row.get("ml_score", 0.0)) >= policy.ml_confirm_threshold
            deep_ok = float(row.get("deep_score", 0.0)) >= policy.deep_confirm_threshold
            hybrid_ok = float(row.get("hybrid_score", 0.0)) >= policy.hybrid_confirm_threshold

            confirmed = corr_ok or ((price_ok and liq_ok) and (ml_ok or deep_ok or hybrid_ok))
            if not confirmed:
                continue

            alerts.append(
                {
                    "timestamp": row["timestamp"],
                    "market_id": str(mid),
                    "p": float(row["p"]),
                    "anomaly_score": float(row["anomaly_score"]),
                    "base_score": float(row.get("base_score", 0.0)),
                    "ml_score": float(row.get("ml_score", 0.0)),
                    "deep_score": float(row.get("deep_score", 0.0)),
                    "hybrid_score": float(row.get("hybrid_score", 0.0)),
                    "explanation": _explain(row),
                }
            )

    return pd.DataFrame(alerts)


# Alias (in case main.py or old code expects this name)
def score_polymarket_anomalies(df: pd.DataFrame, cfg: DetectorConfig) -> pd.DataFrame:
    return compute_market_scores(df, cfg=cfg)
