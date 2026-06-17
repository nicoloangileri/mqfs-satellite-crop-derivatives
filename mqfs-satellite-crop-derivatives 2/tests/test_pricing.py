"""
Pricing-engine tests.

Covers the economic invariants of the premium, the analytic (Bachelier)
validation of the Monte Carlo, convergence, and the three-way agreement of the
NumPy / C / C++ engines to within Monte Carlo standard error.

The native-engine tests skip cleanly (rather than fail) when the shared
libraries have not been compiled, so ``pytest`` is green on a fresh checkout.
"""
import numpy as np
import pytest

from pricing_engine.payoff import Contract, strike_from_quantile, payoff, aggregate_index
from pricing_engine.pricer import MarketParams, price_contract
from pricing_engine.monte_carlo import MCConfig, simulate_paths


SEASONAL = np.array([0.55, 0.62, 0.68, 0.71, 0.70, 0.64, 0.55, 0.45])
KAPPA, SIGMA, DT = 6.0, 0.12, 10.0 / 365.0


def _market(theta=0.25, lam=0.15):
    return MarketParams(risk_free=0.03, horizon_years=0.5,
                        sd_loading_theta=theta, wang_lambda=lam)


# --------------------------------------------------------------------------- #
# Economic invariants
# --------------------------------------------------------------------------- #
def test_premium_monotonic_in_strike():
    mc = MCConfig(n_paths=120_000, seed=1)
    mkt = _market()
    strikes = [0.45, 0.52, 0.58, 0.64]
    fairs = []
    for k in strikes:
        c = Contract(strike=k, tick=250_000, limit=50_000)
        fairs.append(price_contract(SEASONAL, KAPPA, SIGMA, DT, c, mkt, mc=mc).fair_value)
    # higher floor => protection triggers more often => richer premium
    assert all(b >= a - 1e-6 for a, b in zip(fairs, fairs[1:]))


def test_zero_loading_equals_fair_value():
    mc = MCConfig(n_paths=120_000, seed=2)
    c = Contract(strike=0.58, tick=250_000, limit=50_000)
    res = price_contract(SEASONAL, KAPPA, SIGMA, DT, c, _market(theta=0.0, lam=0.0), mc=mc)
    assert np.isclose(res.sd_premium, res.fair_value, rtol=1e-9)
    # Wang with lambda=0 is the identity distortion -> also the fair value
    assert np.isclose(res.wang_premium, res.fair_value, rtol=0.02)


def test_risk_loading_raises_premium():
    mc = MCConfig(n_paths=120_000, seed=3)
    c = Contract(strike=0.58, tick=250_000, limit=50_000)
    res = price_contract(SEASONAL, KAPPA, SIGMA, DT, c, _market(theta=0.25, lam=0.15), mc=mc)
    assert res.sd_premium > res.fair_value
    assert res.wang_premium > res.fair_value


def test_digital_payout_bounded_by_limit():
    mc = MCConfig(n_paths=80_000, seed=4)
    c = Contract(strike=0.58, tick=250_000, limit=50_000, style="digital")
    res = price_contract(SEASONAL, KAPPA, SIGMA, DT, c, _market(), mc=mc)
    assert 0.0 <= res.expected_payout <= c.limit


def test_strike_from_quantile():
    hist = np.array([0.50, 0.55, 0.60, 0.62, 0.65, 0.58, 0.49, 0.71, 0.66, 0.53])
    k = strike_from_quantile(hist, q=0.20)
    assert np.isclose(k, np.quantile(hist, 0.20))


def test_payoff_capped():
    c = Contract(strike=0.7, tick=250_000, limit=50_000)
    # deep loss should be capped at the limit, not tick*(K-I)
    pay = payoff(np.array([0.1]), c)
    assert np.isclose(pay[0], c.limit)


# --------------------------------------------------------------------------- #
# Analytic validation (Bachelier) of the Monte Carlo
# --------------------------------------------------------------------------- #
def test_uncapped_put_matches_bachelier():
    """MC mean of tick*max(0,K-I) must match the analytic Gaussian shortfall."""
    mc = MCConfig(n_paths=400_000, seed=5)
    paths = simulate_paths(SEASONAL, KAPPA, SIGMA, DT, cfg=mc)
    index = aggregate_index(paths, how="mean", dt=DT)
    K = 0.62
    mc_mean = np.mean(np.maximum(0.0, K - index))

    from scipy.stats import norm
    m, s = index.mean(), index.std(ddof=1)
    d = (K - m) / s
    analytic = (K - m) * norm.cdf(d) + s * norm.pdf(d)

    se = np.std(np.maximum(0.0, K - index)) / np.sqrt(mc.n_paths)
    assert abs(mc_mean - analytic) < 5.0 * se


def test_mc_convergence_across_seeds():
    """Fair value should be stable across seeds to within a few standard errors."""
    c = Contract(strike=0.58, tick=250_000, limit=50_000)
    mkt = _market()
    vals = []
    for sd in (10, 20, 30, 40):
        mc = MCConfig(n_paths=200_000, seed=sd)
        vals.append(price_contract(SEASONAL, KAPPA, SIGMA, DT, c, mkt, mc=mc).fair_value)
    vals = np.array(vals)
    # spread across seeds is small relative to the level
    assert vals.std() / vals.mean() < 0.05


# --------------------------------------------------------------------------- #
# Native engines: NumPy vs C vs C++ to within Monte Carlo standard error
# --------------------------------------------------------------------------- #
def _native_or_skip(fn):
    """Call a native pricer on a tiny contract; skip the test if not built."""
    try:
        c = Contract(strike=0.58, tick=250_000, limit=50_000)
        return fn(SEASONAL, KAPPA, SIGMA, DT, c, _market(), mc=MCConfig(n_paths=2000, seed=1))
    except FileNotFoundError as e:  # shared object not compiled on this machine
        pytest.skip(f"native engine not built: {e}")


def test_c_engine_matches_numpy():
    from bindings.c_pricer_wrapper import price_contract_c
    _native_or_skip(price_contract_c)

    mc = MCConfig(n_paths=400_000, seed=99)
    mkt = _market()
    c = Contract(strike=0.58, tick=250_000, limit=50_000)
    py = price_contract(SEASONAL, KAPPA, SIGMA, DT, c, mkt, mc=mc)
    cc = price_contract_c(SEASONAL, KAPPA, SIGMA, DT, c, mkt, mc=mc)

    disc = np.exp(-mkt.risk_free * mkt.horizon_years)
    se = disc * py.payout_std / np.sqrt(mc.n_paths)
    assert abs(cc.fair_value - py.fair_value) < 4.0 * se
    assert abs(cc.trigger_probability - py.trigger_probability) < 0.01


def test_cpp_engine_matches_numpy():
    from bindings.cpp_pricer_wrapper import price_contract_cpp
    _native_or_skip(price_contract_cpp)

    mc = MCConfig(n_paths=400_000, seed=99)
    mkt = _market()
    c = Contract(strike=0.58, tick=250_000, limit=50_000)
    py = price_contract(SEASONAL, KAPPA, SIGMA, DT, c, mkt, mc=mc)
    cp = price_contract_cpp(SEASONAL, KAPPA, SIGMA, DT, c, mkt, mc=mc)

    disc = np.exp(-mkt.risk_free * mkt.horizon_years)
    se = disc * py.payout_std / np.sqrt(mc.n_paths)
    assert abs(cp.fair_value - py.fair_value) < 4.0 * se
    assert abs(cp.trigger_probability - py.trigger_probability) < 0.01
