"""
pricing_engine.ou_calibration
==============================
Calibrate the stochastic model for the NDVI anomaly:

    dX_t = -kappa * X_t dt + sigma dW_t          (Ornstein-Uhlenbeck, reverts to 0)

Discretely sampled at step dt this is an AR(1):

    X_{t+1} = a * X_t + eps_t,   a = exp(-kappa*dt),
    Var(eps) = sigma^2 * (1 - exp(-2*kappa*dt)) / (2*kappa)

so the exact MLE has a closed form (OLS of X_{t+1} on X_t). We expose that fast
estimator plus an optional ``scipy.optimize`` cross-check, and we also fit the
seasonal climatology so the engine knows the level the anomaly reverts around.

The anomaly is mean-reverting and (loosely) bounded, which is exactly why GBM —
the workhorse of the oil paper for a *traded* asset — is the wrong model for a
*biophysical* index. This is the central modelling argument of the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from quant_layer.feature_engineering import Climatology, fit_climatology  # re-export


@dataclass
class OUParams:
    kappa: float          # mean-reversion speed (per year)
    sigma: float          # diffusion volatility (per sqrt-year)
    dt: float             # sampling step (years)
    x0: float = 0.0       # current anomaly state (initial condition for MC)
    half_life_years: float = float("nan")

    def __post_init__(self):
        if self.kappa > 0:
            self.half_life_years = np.log(2.0) / self.kappa


def calibrate_ou(anomaly: np.ndarray, dt: float) -> OUParams:
    """Closed-form (OLS/MLE) calibration of the OU anomaly process.

    Parameters
    ----------
    anomaly : 1-D array
        De-seasonalised NDVI residual (NaNs dropped internally).
    dt : float
        Sampling step in years (e.g. 10/365 for dekadal data).
    """
    x = np.asarray(anomaly, dtype=float)
    x = x[np.isfinite(x)]
    x0, x1 = x[:-1], x[1:]

    # OLS: x1 = a*x0 + c  (intercept c absorbs any residual level; ~0 for anomalies)
    A = np.column_stack([x0, np.ones_like(x0)])
    (a, c), *_ = np.linalg.lstsq(A, x1, rcond=None)
    a = float(np.clip(a, 1e-6, 0.999999))   # keep stationary & invertible

    resid = x1 - (a * x0 + c)
    var_eps = float(np.var(resid, ddof=2))

    kappa = -np.log(a) / dt
    sigma = np.sqrt(var_eps * 2.0 * kappa / (1.0 - a ** 2))
    return OUParams(kappa=float(kappa), sigma=float(sigma), dt=dt, x0=float(x[-1]))


def calibrate_ou_mle(anomaly: np.ndarray, dt: float) -> OUParams:
    """Numerical MLE cross-check (Gaussian transition density). Same target as
    :func:`calibrate_ou`; used in tests to confirm the closed form."""
    from scipy.optimize import minimize
    x = np.asarray(anomaly, dtype=float)
    x = x[np.isfinite(x)]
    x0, x1 = x[:-1], x[1:]

    def nll(theta):
        kappa, log_sigma = theta
        sigma = np.exp(log_sigma)
        if kappa <= 0:
            return 1e12
        a = np.exp(-kappa * dt)
        var = sigma ** 2 * (1 - a ** 2) / (2 * kappa)
        mu = a * x0
        return 0.5 * np.sum(np.log(2 * np.pi * var) + (x1 - mu) ** 2 / var)

    seed = calibrate_ou(anomaly, dt)
    res = minimize(nll, x0=[seed.kappa, np.log(seed.sigma)], method="Nelder-Mead")
    kappa, log_sigma = res.x
    return OUParams(kappa=float(kappa), sigma=float(np.exp(log_sigma)), dt=dt, x0=seed.x0)


def calibrate_full(features, dt: float, harmonics: int = 2):
    """Convenience: (re)fit climatology + OU from a feature frame.

    The OU driver is calibrated on the *raw* de-seasonalised residual
    (``anomaly_raw``) rather than the smoothed anomaly: smoothing removes the
    genuine variance the stochastic term must reproduce, so calibrating on it
    would understate dispersion and drive the model-implied trigger probability
    to zero. Falls back to ``anomaly`` if the raw column is absent.

    Returns ``(Climatology, OUParams)``.
    """
    clim = features.attrs.get("climatology")
    if clim is None:
        clim = fit_climatology(features.index, features["ndvi_smooth"].to_numpy(), harmonics)
    col = "anomaly_raw" if "anomaly_raw" in features.columns else "anomaly"
    ou = calibrate_ou(features[col].to_numpy(), dt)
    return clim, ou
