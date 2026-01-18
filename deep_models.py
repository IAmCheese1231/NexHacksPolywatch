from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------
# config
# ---------------------------

@dataclass(frozen=True)
class DeepConfig:
    device: str = "cpu"
    # windowing
    seq_len: int = 96          # e.g. 96 * 5min = 8 hours
    stride: int = 1
    train_frac: float = 0.8
    # training
    batch_size: int = 128
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # model sizes
    d_model: int = 32
    dropout: float = 0.1


# ---------------------------
# utilities
# ---------------------------

def _to_float_series(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").astype("float64").to_numpy()
    return x


def _make_windows(x: np.ndarray, L: int, stride: int) -> np.ndarray:
    # x: (T,)
    T = x.shape[0]
    if T < L:
        return np.empty((0, L), dtype=np.float32)
    n = 1 + (T - L) // stride
    out = np.empty((n, L), dtype=np.float32)
    j = 0
    i = 0
    while j < n:
        out[j, :] = x[i:i+L]
        j += 1
        i += stride
    return out


def _robust_z(x: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, float, float]:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    z = (x - med) / (1.4826 * mad)
    return z.astype(np.float32), float(med), float(mad)


def _winsorize(z: np.ndarray, clip: float = 8.0) -> np.ndarray:
    return np.clip(z, -clip, clip).astype(np.float32)


def _minmax01(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    lo = np.min(a)
    hi = np.max(a)
    if hi - lo < eps:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo + eps)


# ---------------------------
# models
# ---------------------------

class TinyMLPAE(nn.Module):
    """Window-level autoencoder: reconstruct the whole window."""
    def __init__(self, L: int, hidden: int, dropout: float):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(L, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(hidden // 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, L),
        )

    def forward(self, x):
        z = self.enc(x)
        y = self.dec(z)
        return y


class TinyGRUForecast(nn.Module):
    """Forecast next value from previous L values."""
    def __init__(self, d_in: int, d_h: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(input_size=d_in, hidden_size=d_h, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_h, 1)

    def forward(self, x):  # x: (B,L,1)
        h, _ = self.gru(x)
        last = h[:, -1, :]
        y = self.head(self.drop(last))
        return y[:, 0]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,max_len,d_model)

    def forward(self, x):  # (B,L,d_model)
        L = x.shape[1]
        return x + self.pe[:, :L, :]


class TinyTransformerForecast(nn.Module):
    """Forecast next value using a TransformerEncoder."""
    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, dim_feedforward=4*d_model
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B,L,1)
        h = self.inp(x)
        h = self.pos(h)
        h = self.enc(h)
        y = self.head(h[:, -1, :])
        return y[:, 0]


# ---------------------------
# train + score
# ---------------------------

def _train_mse(model: nn.Module, dl: DataLoader, cfg: DeepConfig) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    for _ in range(cfg.epochs):
        for (xb,) in dl:
            xb = xb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            yb = model(xb)
            loss = loss_fn(yb, xb)
            loss.backward()
            opt.step()


def _train_forecast(model: nn.Module, dl: DataLoader, cfg: DeepConfig) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    for _ in range(cfg.epochs):
        for xb, yb in dl:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()


def _window_to_point_scores(win_scores: np.ndarray, T: int, L: int, stride: int) -> np.ndarray:
    # Convert window scores to per-timestamp scores by accumulating to the window end.
    out = np.zeros(T, dtype=np.float32)
    cnt = np.zeros(T, dtype=np.float32)
    n = win_scores.shape[0]
    j = 0
    i = 0
    while j < n:
        end = i + L - 1
        out[end] += float(win_scores[j])
        cnt[end] += 1.0
        j += 1
        i += stride
    cnt = np.maximum(cnt, 1.0)
    return out / cnt


def score_deep_models_for_market(
    x_raw: np.ndarray, cfg: DeepConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x_raw: (T,) float series (e.g., logit_p or returns)
    returns: (T,) scores in [0,1] for AE, GRU forecast, Transformer forecast
    """
    T = x_raw.shape[0]
    if T < cfg.seq_len + 2:
        z = np.zeros(T, dtype=np.float32)
        return z, z, z

    # robust normalize (key TSAD preproc pattern) :contentReference[oaicite:2]{index=2}
    z, _, _ = _robust_z(x_raw)
    z = _winsorize(z, 8.0)

    W = _make_windows(z, cfg.seq_len, cfg.stride)  # (N,L)
    if W.shape[0] < 10:
        zz = np.zeros(T, dtype=np.float32)
        return zz, zz, zz

    # split train/test by time
    n = W.shape[0]
    n_train = max(1, int(cfg.train_frac * n))
    W_train = W[:n_train]
    W_all = W

    # ---------------- AE (reconstruction error) ----------------
    ae = TinyMLPAE(L=cfg.seq_len, hidden=max(64, cfg.d_model * 4), dropout=cfg.dropout).to(cfg.device)
    dl_ae = DataLoader(TensorDataset(torch.from_numpy(W_train)), batch_size=cfg.batch_size, shuffle=True)
    _train_mse(ae, dl_ae, cfg)

    ae.eval()
    with torch.no_grad():
        xb = torch.from_numpy(W_all).to(cfg.device)
        yb = ae(xb).cpu().numpy()
    ae_win = np.mean((W_all - yb) ** 2, axis=1).astype(np.float32)
    ae_point = _window_to_point_scores(ae_win, T, cfg.seq_len, cfg.stride)
    ae_01 = _minmax01(ae_point)

    # ---------------- GRU forecast (residual error) ----------------
    # build (X, y) where y is next point after window
    y = z[cfg.seq_len:]
    X = _make_windows(z[:-1], cfg.seq_len, cfg.stride)  # windows ending at t-1
    y = y[:X.shape[0]].astype(np.float32)

    n2 = X.shape[0]
    n2_train = max(1, int(cfg.train_frac * n2))
    Xtr = X[:n2_train]
    ytr = y[:n2_train]

    Xtr_t = torch.from_numpy(Xtr).unsqueeze(-1)  # (N,L,1)
    ytr_t = torch.from_numpy(ytr)                # (N,)
    dl_gru = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=cfg.batch_size, shuffle=True)

    gru = TinyGRUForecast(d_in=1, d_h=cfg.d_model, dropout=cfg.dropout).to(cfg.device)
    _train_forecast(gru, dl_gru, cfg)

    gru.eval()
    with torch.no_grad():
        Xall_t = torch.from_numpy(X).unsqueeze(-1).to(cfg.device)
        pred = gru(Xall_t).cpu().numpy()
    resid = (y - pred).astype(np.float32)
    resid2 = (resid ** 2).astype(np.float32)
    # align to timestamps: resid corresponds to t = seq_len.. end
    gru_point = np.zeros(T, dtype=np.float32)
    gru_point[cfg.seq_len:cfg.seq_len + resid2.shape[0]] = resid2
    gru_01 = _minmax01(gru_point)

    # ---------------- Transformer forecast (residual error) ----------------
    trf = TinyTransformerForecast(d_model=cfg.d_model, nhead=4, num_layers=2, dropout=cfg.dropout).to(cfg.device)
    dl_trf = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=cfg.batch_size, shuffle=True)
    _train_forecast(trf, dl_trf, cfg)

    trf.eval()
    with torch.no_grad():
        pred2 = trf(Xall_t).cpu().numpy()
    resid_t = (y - pred2).astype(np.float32)
    resid2_t = (resid_t ** 2).astype(np.float32)
    trf_point = np.zeros(T, dtype=np.float32)
    trf_point[cfg.seq_len:cfg.seq_len + resid2_t.shape[0]] = resid2_t
    trf_01 = _minmax01(trf_point)

    return ae_01, gru_01, trf_01


def score_deep_models(
    df: pd.DataFrame,
    cfg: DeepConfig,
    value_col: str = "logit_p",
) -> pd.DataFrame:
    """
    Adds:
      ae_score_01, gru_forecast_score_01, trf_forecast_score_01
    """
    out = df.copy()
    out["ae_score_01"] = 0.0
    out["gru_forecast_score_01"] = 0.0
    out["trf_forecast_score_01"] = 0.0

    # per-market training/scoring
    for mid, g in out.groupby("market_id", sort=False):
        gg = g.sort_values("timestamp")
        x = _to_float_series(gg[value_col])
        ae_s, gru_s, trf_s = score_deep_models_for_market(x, cfg)

        out.loc[gg.index, "ae_score_01"] = ae_s
        out.loc[gg.index, "gru_forecast_score_01"] = gru_s
        out.loc[gg.index, "trf_forecast_score_01"] = trf_s

    return out
