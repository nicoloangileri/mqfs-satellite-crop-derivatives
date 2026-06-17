"""
quant_layer.volatility_matrix
=============================
Translate per-zone anomaly series into the structured **volatility matrix**
the desk reasons about:

  * term structure of anomaly volatility (rolling, seasonal);
  * the cross-zone covariance / correlation matrix that governs the *spatial
    basis risk* of a multi-contract book — a portfolio of single-zone parametric
    contracts is only as diversified as this matrix allows.

Optionally we apply Random-Matrix-Theory (Marchenko-Pastur) denoising to the
correlation matrix before it feeds portfolio aggregation, mirroring the RMT
filtering used elsewhere in the MQFS research stack.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def rolling_vol(anomaly: pd.Series, window: int = 9, annualize_factor: float = np.sqrt(36.0)) -> pd.Series:
    """Rolling stdev of the anomaly (36 dekads/yr -> annualisation ~ sqrt(36))."""
    return anomaly.rolling(window, min_periods=max(2, window // 2)).std(ddof=1) * annualize_factor


def seasonal_vol_profile(features: pd.DataFrame, col: str = "anomaly") -> pd.Series:
    """Volatility as a function of day-of-year: when is crop health most uncertain?

    Spring (grain fill) typically shows the highest anomaly volatility — exactly
    the window the contract covers, which is why a flat unconditional vol
    under-prices the risk (the same lesson as the oil paper's vol-clustering).
    """
    doy = features.index.dayofyear
    return features.groupby(doy)[col].std(ddof=1).rename("seasonal_vol")


def build_anomaly_matrix(features_by_zone: Dict[str, pd.DataFrame], col: str = "anomaly") -> pd.DataFrame:
    """Align every zone's anomaly onto a common date index -> wide matrix.

    Columns = zones, rows = dekads. Inner join keeps only co-observed dates so
    the covariance is computed on a balanced panel.
    """
    cols = {zid: f[col] for zid, f in features_by_zone.items()}
    return pd.DataFrame(cols).dropna(how="any")


def covariance(anomaly_matrix: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    cov = anomaly_matrix.cov()
    if annualize:
        cov = cov * 36.0   # dekadal -> annual
    return cov


def correlation(anomaly_matrix: pd.DataFrame) -> pd.DataFrame:
    return anomaly_matrix.corr()


# --------------------------------------------------------------------------- #
# RMT denoising (Marchenko-Pastur)
# --------------------------------------------------------------------------- #
def rmt_denoise(corr: pd.DataFrame, T: int, N: int | None = None) -> pd.DataFrame:
    """Clip eigenvalues below the Marchenko-Pastur upper edge to their mean.

    Removes sampling noise from the empirical correlation matrix before it is
    used for portfolio risk. ``T`` = number of observations, ``N`` = assets.
    """
    N = N or corr.shape[0]
    vals, vecs = np.linalg.eigh(corr.to_numpy())
    q = T / N
    lambda_plus = (1 + np.sqrt(1 / q)) ** 2

    noise = vals < lambda_plus
    if noise.any():
        vals[noise] = vals[noise].mean()       # replace bulk with its average
    clean = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(clean))
    clean = clean / np.outer(d, d)              # renormalise to unit diagonal
    return pd.DataFrame(clean, index=corr.index, columns=corr.columns)


def portfolio_anomaly_vol(cov: pd.DataFrame, weights: np.ndarray | None = None) -> float:
    """sqrt(w' Σ w) — book-level anomaly volatility under exposure weights ``w``."""
    n = cov.shape[0]
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)
    return float(np.sqrt(w @ cov.to_numpy() @ w))
