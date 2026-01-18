from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -------------------------
# Utilities (vector-safe)
# -------------------------

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid for numpy arrays.
    Avoids overflow warnings by clipping.
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _err_to_01(err: np.ndarray) -> np.ndarray:
    """
    Convert reconstruction error to [0,1] anomaly score via robust z + sigmoid.
    """
    err = np.asarray(err, dtype=float)
    if err.size == 0:
        return err
    med = np.median(err)
    mad = np.median(np.abs(err - med)) + 1e-9
    z = (err - med) / (1.4826 * mad)
    return np.clip(_sigmoid_np(z - 1.0), 0.0, 1.0)


# -------------------------
# Model
# -------------------------

class _MLPEncoder(nn.Module):
    def __init__(self, d_in: int, d_latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, d_latent),
        )

    def forward(self, x):
        return self.net(x)


class _MLPDecoder(nn.Module):
    def __init__(self, d_latent: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, d_out),
        )

    def forward(self, z):
        return self.net(z)


class _AutoEncoder(nn.Module):
    def __init__(self, d_in: int, d_latent: int):
        super().__init__()
        self.enc = _MLPEncoder(d_in, d_latent)
        self.dec = _MLPDecoder(d_latent, d_in)

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z)
        return xhat, z


# -------------------------
# Config + Detector
# -------------------------

@dataclass
class AEConfig:
    train_window: int = 5000
    latent_dim: int = 8
    hidden_noise: bool = True
    denoise_std: float = 0.05

    epochs: int = 1
    batch_size: int = 256
    lr: float = 1e-3

    device: str = "cpu"


class AutoencoderDetector:
    def __init__(self, cfg: AEConfig):
        self.cfg = cfg
        self.model: Optional[_AutoEncoder] = None
        self.mu: Optional[np.ndarray] = None
        self.sig: Optional[np.ndarray] = None

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        mu = np.nanmean(X, axis=0)
        sig = np.nanstd(X, axis=0)
        sig = np.where(sig < 1e-9, 1.0, sig)
        self.mu = mu
        self.sig = sig
        return (np.nan_to_num(X, nan=0.0) - mu) / sig

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.sig is not None
        return (np.nan_to_num(X, nan=0.0) - self.mu) / self.sig

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> None:
        if len(df) > self.cfg.train_window:
            train_df = df.iloc[-self.cfg.train_window :].copy()
        else:
            train_df = df.copy()

        X = train_df[feature_cols].to_numpy(dtype=float)
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]
        if X.shape[0] < max(400, 10 * len(feature_cols)):
            self.model = None
            self.mu = None
            self.sig = None
            return

        Xs = self._standardize_fit(X)

        device = torch.device(self.cfg.device)
        self.model = _AutoEncoder(d_in=Xs.shape[1], d_latent=self.cfg.latent_dim).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        loss_fn = nn.MSELoss(reduction="mean")

        Xt = torch.tensor(Xs, dtype=torch.float32)
        ds = TensorDataset(Xt)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)

        self.model.train()
        for _ in range(self.cfg.epochs):
            for (xb,) in dl:
                xb = xb.to(device)

                if self.cfg.hidden_noise:
                    noise = torch.randn_like(xb) * float(self.cfg.denoise_std)
                    xin = xb + noise
                else:
                    xin = xb

                xhat, _ = self.model(xin)
                loss = loss_fn(xhat, xb)

                opt.zero_grad()
                loss.backward()
                opt.step()

        self.model.eval()

    @torch.no_grad()
    def score_and_latent(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["ae_score_01"] = 0.0

        if self.model is None or self.mu is None or self.sig is None:
            # still return placeholder latent columns for downstream code consistency
            for i in range(8):
                out[f"z{i}"] = 0.0
            return out

        X = df[feature_cols].to_numpy(dtype=float)
        mask = np.isfinite(X).all(axis=1)

        Xs = self._standardize_apply(X)
        device = torch.device(self.cfg.device)

        Xt = torch.tensor(Xs, dtype=torch.float32, device=device)
        xhat, z = self.model(Xt)

        err = torch.mean((xhat - Xt) ** 2, dim=1).detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()

        # scores only where mask true
        s01 = _err_to_01(err)
        s_full = np.zeros(len(df), dtype=float)
        s_full[mask] = s01[mask] if s01.shape[0] == len(df) else s01

        out["ae_score_01"] = s_full

        # latent dims
        k = z_np.shape[1]
        for j in range(k):
            col = np.zeros(len(df), dtype=float)
            col[mask] = z_np[mask, j]
            out[f"z{j}"] = col

        return out
