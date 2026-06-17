"""
quant_layer.feature_engineering
================================
Turn a raw, gappy dekadal NDVI series into clean, de-seasonalised signal
ready for the pricing engine.

Pipeline
--------
1. Regularise to a fixed dekadal grid.
2. Gap-fill + smooth (Whittaker or Savitzky-Golay) — clouds leave NaNs that
   must be reconstructed before any moving-average logic.
3. Fit a day-of-year **harmonic climatology** (the deterministic seasonal mean
   s(doy) the OU model reverts toward).
4. Compute the **anomaly** (residual), its z-score, and the Vegetation
   Condition Index (VCI).

Everything is pandas/numpy/scipy only.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter


# --------------------------------------------------------------------------- #
# 1-2. Smoothing / gap-filling
# --------------------------------------------------------------------------- #
def whittaker_smooth(y: np.ndarray, lmbda: float = 1500.0, d: int = 2) -> np.ndarray:
    """Whittaker-Eilers smoother with NaN-aware weights.

    Minimises  ||W(z - y)||^2 + lambda ||D^d z||^2 . Missing samples get weight 0
    so they are *interpolated* by the smoothness penalty — the standard, robust
    way to reconstruct cloud-gapped NDVI phenology.
    """
    y = np.asarray(y, dtype=float)
    m = y.size
    w = np.isfinite(y).astype(float)
    y_filled = np.where(np.isfinite(y), y, 0.0)
    D = sparse.eye(m, format="csc")
    for _ in range(d):
        D = D[1:] - D[:-1]
    W = sparse.diags(w, 0, format="csc")
    A = W + lmbda * (D.T @ D)
    z = spsolve(A.tocsc(), W @ y_filled)
    return z


def smooth_series(s: pd.Series, method: str = "whittaker",
                  lmbda: float = 1500.0, window: int = 5, poly: int = 2) -> pd.Series:
    """Smooth/gap-fill a NDVI Series indexed by date."""
    if method == "whittaker":
        z = whittaker_smooth(s.to_numpy(), lmbda=lmbda)
    elif method == "savgol":
        filled = s.interpolate(limit_direction="both").to_numpy()
        win = window if window % 2 == 1 else window + 1
        z = savgol_filter(filled, window_length=min(win, len(filled) // 2 * 2 + 1), polyorder=poly)
    else:
        raise ValueError(f"unknown smoothing method {method!r}")
    return pd.Series(z, index=s.index, name=s.name)


# --------------------------------------------------------------------------- #
# 3. Harmonic day-of-year climatology
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class Climatology:
    """Fitted harmonic seasonal mean s(doy) = a0 + Σ a_k cos + b_k sin.

    ``eq=False`` keeps identity-based equality: an instance is stored in
    ``DataFrame.attrs`` and pandas compares ``attrs`` during some operations
    (e.g. groupby); the default dataclass ``__eq__`` would compare the numpy
    ``coef`` array element-wise and raise an ambiguous-truth-value error.
    """
    coef: np.ndarray
    harmonics: int

    def predict(self, doy: np.ndarray) -> np.ndarray:
        return _harmonic_design(np.asarray(doy), self.harmonics) @ self.coef


def _harmonic_design(doy: np.ndarray, harmonics: int) -> np.ndarray:
    cols = [np.ones_like(doy, dtype=float)]
    for k in range(1, harmonics + 1):
        ang = 2 * np.pi * k * doy / 365.25
        cols += [np.cos(ang), np.sin(ang)]
    return np.column_stack(cols)


def fit_climatology(dates: pd.DatetimeIndex, values: np.ndarray, harmonics: int = 2) -> Climatology:
    """Least-squares harmonic fit of the seasonal mean."""
    doy = dates.dayofyear.to_numpy()
    X = _harmonic_design(doy, harmonics)
    coef, *_ = np.linalg.lstsq(X, np.asarray(values, dtype=float), rcond=None)
    return Climatology(coef=coef, harmonics=harmonics)


# --------------------------------------------------------------------------- #
# 4. Anomalies, z-scores, VCI
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame, ndvi_col: str = "NDVI",
                   smoothing: dict | None = None, harmonics: int = 2,
                   standardize: bool = True) -> pd.DataFrame:
    """Full feature frame for one zone.

    Returns columns:
        ndvi_raw, ndvi_smooth, seasonal, anomaly, anomaly_z, vci
    indexed by date.
    """
    smoothing = smoothing or {}
    s = df.set_index("date")[ndvi_col]
    smooth = smooth_series(s, **smoothing)

    clim = fit_climatology(s.index, smooth.to_numpy(), harmonics=harmonics)
    seasonal = pd.Series(clim.predict(s.index.dayofyear.to_numpy()), index=s.index)

    anomaly = smooth - seasonal
    # Raw de-seasonalised residual. The smoothed anomaly above is the right
    # object for signal extraction and VCI, but it is the WRONG object to
    # calibrate the stochastic driver on: the Whittaker penalty removes most of
    # the genuine high-frequency variance, so an OU fitted to it collapses to a
    # near-deterministic process and the model-implied trigger probability falls
    # to zero. The raw residual preserves the true dispersion of the index and
    # is what `ou_calibration.calibrate_full` / the pipeline calibrate on.
    anomaly_raw = s - seasonal
    z = (anomaly - anomaly.mean()) / anomaly.std(ddof=1) if standardize else anomaly

    # VCI (Kogan 1995): where today's smoothed NDVI sits between its historical
    # min/max for the same *composite period*. We bucket by dekad-of-year
    # (1..37) rather than exact day-of-year: at 10-day cadence the calendar DOY
    # drifts year to year, so per-DOY buckets would hold a single sample and the
    # min/max normalisation would collapse to a degenerate 0/1. Dekad-of-year
    # gives ~one sample per year per bucket — a stable empirical envelope.
    # 0 = worst on record for that period, 1 = best.
    doy = s.index.dayofyear.to_numpy()
    period = np.minimum((doy - 1) // 10 + 1, 37)
    frame = pd.DataFrame({"ndvi": smooth.to_numpy(), "period": period}, index=s.index)
    grp = frame.groupby("period")["ndvi"]
    dmin = frame["period"].map(grp.min())
    dmax = frame["period"].map(grp.max())
    vci = ((frame["ndvi"] - dmin.values) / (dmax.values - dmin.values + 1e-9)).clip(0, 1)

    out = pd.DataFrame({
        "ndvi_raw": s,
        "ndvi_smooth": smooth,
        "seasonal": seasonal,
        "anomaly": anomaly,
        "anomaly_raw": anomaly_raw,
        "anomaly_z": z,
        "vci": vci.values,
    })
    out.attrs["climatology"] = clim
    return out
