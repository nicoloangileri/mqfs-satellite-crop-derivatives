"""
pricing_engine.monte_carlo
===========================
Reference Monte Carlo simulator of in-season NDVI paths under the
seasonal-mean + Ornstein-Uhlenbeck-anomaly model.

This pure-NumPy engine is the *specification*: the C and C++ engines must agree
with it to within Monte Carlo standard error. Vectorised, with antithetic
variates for variance reduction.

Path model (per dekad k = 1..M, step dt years)
-----------------------------------------------
    X_0 = x0
    X_k = a * X_{k-1} + b * Z_k,     a = exp(-kappa*dt),
                                     b = sigma*sqrt((1-exp(-2*kappa*dt))/(2*kappa))
    NDVI_k = clip(seasonal_k + X_k,  floor, cap)

``seasonal`` is the climatology evaluated on the risk-window day-of-year grid.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class MCConfig:
    n_paths: int = 200_000
    antithetic: bool = True
    seed: int = 20260506
    ndvi_floor: float = -0.20
    ndvi_cap: float = 0.95


def ou_step_constants(kappa: float, sigma: float, dt: float):
    """Exact discretisation constants (a, b) for the OU transition."""
    a = np.exp(-kappa * dt)
    b = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
    return a, b


def simulate_paths(seasonal: np.ndarray, kappa: float, sigma: float, dt: float,
                   x0: float = 0.0, cfg: MCConfig | None = None) -> np.ndarray:
    """Simulate NDVI paths. Returns array of shape (n_paths, M)."""
    cfg = cfg or MCConfig()
    seasonal = np.asarray(seasonal, dtype=float)
    M = seasonal.size
    a, b = ou_step_constants(kappa, sigma, dt)

    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_paths
    if cfg.antithetic:
        half = (n + 1) // 2
        z = rng.standard_normal((half, M))
        Z = np.vstack([z, -z])[:n]           # antithetic pairs
    else:
        Z = rng.standard_normal((n, M))

    X = np.empty((n, M), dtype=float)
    state = np.full(n, x0, dtype=float)
    for k in range(M):
        state = a * state + b * Z[:, k]
        X[:, k] = state

    ndvi = seasonal[None, :] + X
    np.clip(ndvi, cfg.ndvi_floor, cfg.ndvi_cap, out=ndvi)
    return ndvi


def seasonal_grid(climatology, start_doy: int, end_doy: int, step_days: int = 10) -> np.ndarray:
    """Climatology evaluated on the dekadal risk-window day-of-year grid."""
    doy = np.arange(start_doy, end_doy + 1, step_days)
    return climatology.predict(doy)
