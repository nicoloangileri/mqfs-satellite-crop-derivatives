"""
pricing_engine.payoff
======================
Contract definition and payoff mathematics for the satellite-indexed
parametric drought contract.

Settlement
----------
For a season the **settlement index** I is an aggregate of in-window NDVI:
    mean      :  I = (1/M) sum_k NDVI_k          (default)
    integral  :  I = dt * sum_k NDVI_k           (iNDVI / cumulative greenness)
    min       :  I = min_k NDVI_k                 (tail / acute-stress contract)

Payoff (put / "drought floor")
------------------------------
The grower is long protection; payout rises as crop health falls below strike K:

    raw   = tick * max(0, K - I)
    pay   = min(limit, raw)                       (capped)

Digital variant pays the full ``limit`` if I < K (parametric trigger insurance).

Sign convention: payoff to the *insured* (>= 0). The insurer/writer's loss is the
same number; that is the variable whose tail (VaR/CVaR) we report.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Contract:
    strike: float                 # K, NDVI index units
    tick: float                   # EUR per 1.00 NDVI unit below strike
    limit: float                  # EUR payout cap
    index: str = "mean"           # mean | integral | min
    style: str = "put"            # put | digital
    dt: float = 10 / 365.0        # step (years) — needed for the integral index

    def describe(self) -> str:
        return (f"{self.style} on seasonal NDVI[{self.index}] | K={self.strike:.3f} "
                f"tick=EUR{self.tick:,.0f}/NDVI cap=EUR{self.limit:,.0f}")


def aggregate_index(ndvi_paths: np.ndarray, how: str = "mean", dt: float = 10 / 365.0) -> np.ndarray:
    """Collapse simulated NDVI paths (shape [n_paths, M]) to one index per path."""
    if how == "mean":
        return ndvi_paths.mean(axis=1)
    if how == "integral":
        return dt * ndvi_paths.sum(axis=1)
    if how == "min":
        return ndvi_paths.min(axis=1)
    raise ValueError(f"unknown index aggregator {how!r}")


def payoff(index_values: np.ndarray, contract: Contract) -> np.ndarray:
    """Vectorised payoff (EUR) given realised/ simulated index values."""
    I = np.asarray(index_values, dtype=float)
    if contract.style == "put":
        raw = contract.tick * np.maximum(0.0, contract.strike - I)
        return np.minimum(contract.limit, raw)
    if contract.style == "digital":
        return np.where(I < contract.strike, contract.limit, 0.0)
    raise ValueError(f"unknown payoff style {contract.style!r}")


def strike_from_quantile(historical_index: np.ndarray, q: float = 0.20) -> float:
    """Set K at the q-th historical percentile of the seasonal index.

    A 20% strike means the contract triggers in roughly the worst 1-in-5 seasons
    before any risk premium — a transparent, auditable anchoring for growers.
    """
    return float(np.quantile(np.asarray(historical_index, dtype=float), q))
