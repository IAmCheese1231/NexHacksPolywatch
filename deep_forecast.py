from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from utils import sigmoid


@dataclass
class ForecastConfig:
    seq_len: int = 32
    train_window: int = 6000
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    device: str = "cpu"


class GRUForecaster(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=d_in, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, d_in)

    def forward(self, x):
        y, _ = self.gru(x)
        h = y[:, -1, :]
        return self.head(h)


class TransformerForecaster(nn.Module):
    def __init__(self, d_in: int, nhead: int = 4, dmodel: int = 64, nlayers: int = 2):
        super().__init__()
        self.proj = nn.Linear(d_in, dmodel)
        enc_layer = nn.TransformerEncoderLayer(d_model=dmodel, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Linear(dmodel, d_in)

    def forward(self, x):
        h = self.proj(x)
        h = self.enc(h)
        return self.head(h[:, -1, :])


def _make_sequences(X: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    n, d = X.shape
    m = n - seq_len
    if m <= 0:
        return np.zeros((0, seq_len, d), dtype=np.float32), np.zeros((0, d), dtype=np.float32)

    xs = np.zeros((m, seq_len, d), dtype=np.float32)
    ys = np.zeros((m, d), dtype=np.float32)
    for i in range(m):
        xs[i] = X[i : i + seq_len]
        ys[i] = X[i + seq_len]
    return xs, ys


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _err_to_01(err: np.ndarray) -> np.ndarray:
    if err.size == 0:
        return err
    err = np.asarray(err, dtype=float)
    med = np.median(err)
    mad = np.median(np.abs(err - med)) + 1e-9
    z = (err - med) / (1.4826 * mad)
    return np.clip(_sigmoid_np(z - 1.0), 0.0, 1.0)


class SequenceForecastDetector:
    def __init__(self, cfg: ForecastConfig, model_type: str = "gru"):
        self.cfg = cfg
        self.model_type = model_type
        self.model: Optional[nn.Module] = None
        self.mu: Optional[np.ndarray] = None
        self.sig: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        if len(df) > self.cfg.train_window:
            train_df = df.iloc[-self.cfg.train_window :].copy()
        else:
            train_df = df.copy()

        X = train_df[feature_cols].to_numpy(dtype=float)
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]
        if X.shape[0] < (self.cfg.seq_len + 500):
            self.model = None
            return

        mu = np.median(X, axis=0)
        mad = np.median(np.abs(X - mu), axis=0) + 1e-6
        Xn = (X - mu) / (1.4826 * mad)

        self.mu = mu
        self.sig = 1.4826 * mad

        xs, ys = _make_sequences(Xn.astype(np.float32), self.cfg.seq_len)
        if xs.shape[0] < 500:
            self.model = None
            return

        device = torch.device(self.cfg.device)
        d_in = xs.shape[-1]
        if self.model_type == "transformer":
            model = TransformerForecaster(d_in=d_in).to(device)
        else:
            model = GRUForecaster(d_in=d_in).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        loss_fn = nn.MSELoss()

        Xst = torch.tensor(xs, dtype=torch.float32)
        Yst = torch.tensor(ys, dtype=torch.float32)
        n = Xst.shape[0]

        for _ in range(self.cfg.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.cfg.batch_size):
                idx = perm[i : i + self.cfg.batch_size]
                xb = Xst[idx].to(device)
                yb = Yst[idx].to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.model = model

    @torch.no_grad()
    def score(self, df: pd.DataFrame, feature_cols: list[str], score_col: str) -> pd.Series:
        s = pd.Series(0.0, index=df.index, name=score_col)
        if self.model is None or self.mu is None or self.sig is None:
            return s

        X = df[feature_cols].to_numpy(dtype=float)
        ok = np.isfinite(X).all(axis=1)
        if not bool(np.all(ok)):
            return s

        Xn = (X - self.mu) / (self.sig + 1e-9)
        xs, ys = _make_sequences(Xn.astype(np.float32), self.cfg.seq_len)
        if xs.shape[0] == 0:
            return s

        device = torch.device(self.cfg.device)
        Xst = torch.tensor(xs, dtype=torch.float32).to(device)
        pred = self.model(Xst).detach().cpu().numpy()
        err = np.mean((pred - ys) ** 2, axis=1)
        sc = _err_to_01(err)

        s.iloc[self.cfg.seq_len :] = sc
        return s
