import numpy as np
from scipy.stats import pearsonr, spearmanr
import dcor
from statsmodels.tsa.stattools import adfuller
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression


def _clean(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    return a, b

def returns(prices):
    p = np.asarray(prices, dtype=float)
    if len(p) < 3:
        return p
    r = np.diff(np.log(np.clip(p, 1e-9, None)))
    return r

def pearson(a, b):
    a, b = _clean(a, b)
    if len(a) < 10:
        return None
    return float(pearsonr(a, b)[0])

def spearman(a, b):
    a, b = _clean(a, b)
    if len(a) < 10:
        return None
    return float(spearmanr(a, b)[0])

def distance_corr(a, b):
    a, b = _clean(a, b)
    if len(a) < 20:
        return None
    return float(dcor.distance_correlation(a, b))

def mutual_info(x, y, *, n_neighbors=5, random_state=0):
    """
    Mutual information between two continuous series using sklearn's KNN estimator.
    Returns a nonnegative scalar (higher = more dependence).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Drop NaNs pairwise
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 10:
        return 0.0

    # sklearn expects 2D features for X, 1D target y
    mi = mutual_info_regression(
        X=x.reshape(-1, 1),
        y=y,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    return float(max(0.0, mi[0]))
def best_lag_corr(a, b, max_lag=24):
    # cross-correlation over lags (lead/lag detection)
    a, b = _clean(a, b)
    if len(a) < 50:
        return None
    best = (0, -1.0)
    for lag in range(-max_lag, max_lag+1):
        if lag < 0:
            x, y = a[-lag:], b[:len(b)+lag]
        elif lag > 0:
            x, y = a[:-lag], b[lag:]
        else:
            x, y = a, b
        if len(x) < 20:
            continue
        c = pearson(x, y)
        if c is None:
            continue
        if abs(c) > abs(best[1]):
            best = (lag, c)
    return {"lag": int(best[0]), "corr": float(best[1])}

def cointegration_spread_adf(a, b):
    # very lightweight cointegration proxy:
    # spread = a - beta*b where beta = OLS slope; test ADF(spread)
    a, b = _clean(a, b)
    if len(a) < 80:
        return None
    X = np.vstack([b, np.ones(len(b))]).T
    beta, intercept = np.linalg.lstsq(X, a, rcond=None)[0]
    spread = a - (beta*b + intercept)
    # ADF: lower p-value -> more stationary spread -> cointegration evidence
    pval = adfuller(spread, autolag="AIC")[1]
    return {"beta": float(beta), "pvalue": float(pval)}
