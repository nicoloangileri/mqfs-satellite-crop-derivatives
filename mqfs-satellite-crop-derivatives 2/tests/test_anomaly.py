"""Feature-engineering, OU-calibration and anomaly-detection tests."""
import numpy as np
import pandas as pd

from pricing_engine.ou_calibration import calibrate_ou, calibrate_ou_mle, OUParams
from pricing_engine.monte_carlo import ou_step_constants
from quant_layer.anomaly_detection import seasonal_index, detect_episodes, flag_shocks


# --------------------------------------------------------------------------- #
# Feature frame
# --------------------------------------------------------------------------- #
def test_feature_columns_present(features):
    for col in ("ndvi_raw", "ndvi_smooth", "seasonal", "anomaly", "anomaly_z", "vci"):
        assert col in features.columns
    assert "climatology" in features.attrs


def test_anomaly_is_centred(features):
    # de-seasonalised residual should hover around zero
    assert abs(features["anomaly"].mean()) < 0.02


def test_vci_bounded(features):
    v = features["vci"].dropna()
    assert v.min() >= -1e-9 and v.max() <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# OU calibration round-trip — the headline statistical guarantee
# --------------------------------------------------------------------------- #
def _simulate_ou(kappa, sigma, dt, n, seed=0):
    a, b = ou_step_constants(kappa, sigma, dt)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = 0.0
    for k in range(1, n):
        x[k] = a * x[k - 1] + b * z[k]
    return x


def test_ou_roundtrip_recovers_parameters():
    kappa_true, sigma_true, dt = 6.0, 0.12, 10.0 / 365.0
    x = _simulate_ou(kappa_true, sigma_true, dt, n=20000, seed=42)
    est = calibrate_ou(x, dt)
    assert isinstance(est, OUParams)
    # closed-form estimator is consistent; loose bands for finite sample
    assert abs(est.kappa - kappa_true) / kappa_true < 0.15
    assert abs(est.sigma - sigma_true) / sigma_true < 0.10


def test_ou_mle_agrees_with_closed_form():
    kappa_true, sigma_true, dt = 5.0, 0.10, 10.0 / 365.0
    x = _simulate_ou(kappa_true, sigma_true, dt, n=15000, seed=7)
    a = calibrate_ou(x, dt)
    b = calibrate_ou_mle(x, dt)
    assert abs(a.kappa - b.kappa) / a.kappa < 0.05
    assert abs(a.sigma - b.sigma) / a.sigma < 0.05


def test_ou_half_life():
    p = OUParams(kappa=np.log(2.0) / 0.1, sigma=0.1, dt=0.1)
    assert np.isclose(p.half_life_years, 0.1)


# --------------------------------------------------------------------------- #
# Seasonal settlement index + drought episodes
# --------------------------------------------------------------------------- #
def test_seasonal_index_one_value_per_year(features):
    si = seasonal_index(features, 75, 151, how="mean")
    assert si.index.is_unique
    assert (si.between(-0.2, 1.0)).all()


def test_seasonal_index_min_below_mean(features):
    mean_idx = seasonal_index(features, 75, 151, how="mean")
    min_idx = seasonal_index(features, 75, 151, how="min")
    # acute (min) index sits at or below the seasonal mean every year
    assert (min_idx <= mean_idx + 1e-9).all()


def test_drought_years_detected_at_peak(features):
    # the synthetic generator injects droughts in 2017/2020/2023/2026; an
    # episode may begin in the preceding winter trough, so we key on the dekad
    # of *maximum stress* (peak), which is the meaningful drought timing.
    episodes = detect_episodes(features, max_gap=1)
    assert len(episodes) > 0
    peak_years = {e.peak.year for e in episodes}
    injected = {2017, 2020, 2023, 2026}
    assert len(peak_years & injected) >= 3


def test_worst_seasonal_index_years_are_droughts(features):
    # The contract settles on the seasonal index: the worst seasons (lowest
    # mean NDVI over the risk window) must be the injected drought years. This
    # is the property the pricing actually depends on.
    si = seasonal_index(features, 75, 151, how="mean")
    worst4 = set(si.nsmallest(4).index)
    injected = {2017, 2020, 2023, 2026}
    assert len(worst4 & injected) >= 3


def test_flag_shocks_returns_boolean(features):
    shock = flag_shocks(features, zscore_trigger=-1.5)
    assert shock.dtype == bool
    assert len(shock) == len(features)
