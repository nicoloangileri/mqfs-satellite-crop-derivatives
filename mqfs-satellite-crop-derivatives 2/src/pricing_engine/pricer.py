r"""
pricing_engine.pricer
=====================
The pricing orchestrator. Given a calibrated NDVI model and a contract, it
produces a full pricing sheet: fair value, risk-loaded premiums, trigger
statistics, and the writer's tail-risk (VaR / CVaR).

Why NOT textbook risk-neutral pricing
-------------------------------------
NDVI is **not a traded asset**. You cannot build a self-financing hedging
portfolio in "satellite greenness", so the market is *incomplete* and there is
no unique equivalent martingale measure obtained by replication. Pricing a
weather/crop derivative by naive risk-neutral expectation is a category error.

The defensible routes, both implemented here:

1. **Actuarial standard-deviation principle** (transparent; reinsurance market
   standard):
        Premium = exp(-rT) * ( E_P[payoff] + theta * SD_P[payoff] )

2. **Wang transform** (Wang 2000; distortion / market-price-of-risk):
        Premium = exp(-rT) * \int_0^inf  g( S_Y(y) ) dy,
        g(u) = Phi( Phi^{-1}(u) + lambda )
   which loads the loss distribution's tail by the market price of risk lambda
   and is consistent with CAPM/Black-Scholes in the Gaussian-traded limit.

Both reduce to the discounted expected payout when theta = lambda = 0, which we
also report as the **fair (burn) value** — the natural lower bound for a quote.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable
import numpy as np
from scipy.stats import norm

from .payoff import Contract, aggregate_index, payoff
from .monte_carlo import MCConfig, simulate_paths


@dataclass
class MarketParams:
    risk_free: float = 0.03
    horizon_years: float = 0.50
    sd_loading_theta: float = 0.25
    wang_lambda: float = 0.15


@dataclass
class PricingResult:
    fair_value: float            # exp(-rT) E[payoff]  (burn cost, no load)
    sd_premium: float            # standard-deviation principle
    wang_premium: float          # Wang-transform premium
    expected_payout: float       # undiscounted E[payoff]
    trigger_probability: float   # P(index < strike)
    expected_payout_if_triggered: float
    payout_std: float
    var_95: float                # writer's 95% Value-at-Risk of payout
    cvar_95: float               # writer's 95% Conditional VaR (expected shortfall)
    loss_ratio_at_sd: float      # E[payout] / sd_premium  (quote adequacy)
    n_paths: int
    bachelier_check: float       # analytic uncapped E[max(0,K-I)] for validation

    def to_dict(self):
        return asdict(self)

    def pretty(self) -> str:
        return "\n".join([
            f"  Fair (burn) value           EUR {self.fair_value:14,.2f}",
            f"  SD-principle premium        EUR {self.sd_premium:14,.2f}",
            f"  Wang-transform premium      EUR {self.wang_premium:14,.2f}",
            f"  Trigger probability         {100 * self.trigger_probability:14.2f} %",
            f"  E[payout | triggered]       EUR {self.expected_payout_if_triggered:14,.2f}",
            f"  Writer VaR 95%              EUR {self.var_95:14,.2f}",
            f"  Writer CVaR 95%             EUR {self.cvar_95:14,.2f}",
            f"  Loss ratio @ SD premium     {100 * self.loss_ratio_at_sd:14.1f} %",
        ])


# --------------------------------------------------------------------------- #
# Wang transform on the empirical loss distribution
# --------------------------------------------------------------------------- #
def wang_premium(payouts: np.ndarray, lam: float, n_grid: int = 2000) -> float:
    r"""Undiscounted Wang-transform price of a non-negative loss variable.

    H[Y] = \int_0^{ymax} g(Shat(y)) dy,  g(u)=Phi(Phi^{-1}(u)+lam).
    Computed by trapezoidal integration of the distorted empirical survival.
    """
    y = np.asarray(payouts, dtype=float)
    ymax = y.max()
    if ymax <= 0:
        return 0.0
    grid = np.linspace(0.0, ymax, n_grid)
    # empirical survival S(g) = P(Y > g)
    surv = 1.0 - np.searchsorted(np.sort(y), grid, side="right") / y.size
    surv = np.clip(surv, 1e-12, 1.0 - 1e-12)
    g = norm.cdf(norm.ppf(surv) + lam)
    return float(np.trapezoid(g, grid))


def _bachelier_put(mean: float, std: float, strike: float) -> float:
    """Analytic E[max(0, K - I)] for I ~ Normal(mean, std). Validation reference
    for the *uncapped, unscaled* expected shortfall (tick=1, no cap)."""
    if std <= 0:
        return max(0.0, strike - mean)
    d = (strike - mean) / std
    return (strike - mean) * norm.cdf(d) + std * norm.pdf(d)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def price_contract(seasonal: np.ndarray, kappa: float, sigma: float, dt: float,
                   contract: Contract, market: MarketParams,
                   x0: float = 0.0, mc: MCConfig | None = None,
                   path_simulator: Callable | None = None) -> PricingResult:
    """Price one parametric contract by Monte Carlo.

    Parameters
    ----------
    seasonal : array
        Climatology on the risk-window dekadal grid (length M).
    kappa, sigma, dt, x0 : float
        Calibrated OU parameters + current anomaly state.
    contract, market : dataclasses
    path_simulator : callable, optional
        Inject the C/C++ engine here. Must accept the same signature as
        :func:`pricing_engine.monte_carlo.simulate_paths` and return an
        ``(n_paths, M)`` array. Defaults to the NumPy reference engine.
    """
    mc = mc or MCConfig()
    sim = path_simulator or simulate_paths

    ndvi_paths = sim(seasonal, kappa, sigma, dt, x0=x0, cfg=mc)
    index = aggregate_index(ndvi_paths, how=contract.index, dt=dt)
    payouts = payoff(index, contract)

    disc = np.exp(-market.risk_free * market.horizon_years)
    e_pay = float(payouts.mean())
    sd_pay = float(payouts.std(ddof=1))

    triggered = payouts > 0
    p_trig = float(triggered.mean())
    e_if_trig = float(payouts[triggered].mean()) if triggered.any() else 0.0

    var95 = float(np.quantile(payouts, 0.95))
    tail = payouts[payouts >= var95]
    cvar95 = float(tail.mean()) if tail.size else var95

    sd_prem = disc * (e_pay + market.sd_loading_theta * sd_pay)
    wang_prem = disc * wang_premium(payouts, market.wang_lambda)

    # Bachelier reference uses the *uncapped, tick=1* shortfall on the index.
    bach = contract.tick * _bachelier_put(float(index.mean()), float(index.std(ddof=1)),
                                           contract.strike)

    return PricingResult(
        fair_value=disc * e_pay,
        sd_premium=sd_prem,
        wang_premium=wang_prem,
        expected_payout=e_pay,
        trigger_probability=p_trig,
        expected_payout_if_triggered=e_if_trig,
        payout_std=sd_pay,
        var_95=var95,
        cvar_95=cvar95,
        loss_ratio_at_sd=(e_pay / sd_prem) if sd_prem > 0 else float("nan"),
        n_paths=ndvi_paths.shape[0],
        bachelier_check=min(bach, contract.limit),
    )
