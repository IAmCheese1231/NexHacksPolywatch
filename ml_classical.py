from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler


@dataclass
class ClassicalConfig:
    train_window: int = 3000

    iforest_n_estimators: int = 200
    iforest_contamination: float = 0.01

    lof_n_neighbors: int = 35
    lof_contamination: float = 0.01

    ocsvm_nu: float = 0.01
    ocsvm_gamma: str = "scale"

    do_scale: bool = True


def _prep_matrix(df: pd.DataFrame, feature_cols: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=float)
    mask = np.isfinite(X).all(axis=1)
    return X[mask], mask


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _score_to_01(raw: np.ndarray) -> np.ndarray:
    """
    Convert raw anomaly scores (higher = more anomalous) to [0,1] using robust z and sigmoid.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.size == 0:
        return raw

    med = np.median(raw)
    mad = np.median(np.abs(raw - med)) + 1e-9
    z = (raw - med) / (1.4826 * mad)

    # was: sigmoid(z - 1.0) using scalar sigmoid -> crashes on arrays
    return np.clip(_sigmoid_np(z - 1.0), 0.0, 1.0)


class ClassicalDetectors:
    def __init__(self, cfg: ClassicalConfig):
        self.cfg = cfg
        self.scaler: Optional[RobustScaler] = None
        self.iforest: Optional[IsolationForest] = None
        self.lof: Optional[LocalOutlierFactor] = None
        self.ocsvm: Optional[OneClassSVM] = None

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        if len(df) > self.cfg.train_window:
            train_df = df.iloc[-self.cfg.train_window :].copy()
        else:
            train_df = df.copy()

        X, _ = _prep_matrix(train_df, feature_cols)
        if X.shape[0] < max(200, 5 * len(feature_cols)):
            self.scaler = None
            self.iforest = None
            self.lof = None
            self.ocsvm = None
            return

        if self.cfg.do_scale:
            self.scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(10.0, 90.0))
            Xs = self.scaler.fit_transform(X)
        else:
            self.scaler = None
            Xs = X

        self.iforest = IsolationForest(
            n_estimators=self.cfg.iforest_n_estimators,
            contamination=self.cfg.iforest_contamination,
            random_state=0,
        )
        self.iforest.fit(Xs)

        self.lof = LocalOutlierFactor(
            n_neighbors=self.cfg.lof_n_neighbors,
            contamination=self.cfg.lof_contamination,
            novelty=True,
        )
        self.lof.fit(Xs)

        self.ocsvm = OneClassSVM(nu=self.cfg.ocsvm_nu, gamma=self.cfg.ocsvm_gamma)
        self.ocsvm.fit(Xs)

    def score(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["iforest_score_01"] = 0.0
        out["lof_score_01"] = 0.0
        out["ocsvm_score_01"] = 0.0

        if self.iforest is None or self.lof is None or self.ocsvm is None:
            return out

        X, mask = _prep_matrix(df, feature_cols)
        if X.shape[0] == 0:
            return out

        if self.scaler is not None:
            Xs = self.scaler.transform(X)
        else:
            Xs = X

        # NOTE: sklearn scores: higher -> more normal; we negate to make higher -> more anomalous
        if_raw = -self.iforest.score_samples(Xs)
        lof_raw = -self.lof.score_samples(Xs)
        oc_raw = -self.ocsvm.decision_function(Xs).ravel()

        if01 = _score_to_01(if_raw)
        lof01 = _score_to_01(lof_raw)
        oc01 = _score_to_01(oc_raw)

        idxs = np.where(mask)[0]
        out.iloc[idxs, out.columns.get_loc("iforest_score_01")] = if01
        out.iloc[idxs, out.columns.get_loc("lof_score_01")] = lof01
        out.iloc[idxs, out.columns.get_loc("ocsvm_score_01")] = oc01
        return out
