"""
oracle.synthetic
================
Realistic synthetic NDVI/NDMI generator. Lets the *entire* pipeline
(quant layer + pricing engine) run end-to-end with no GEE credentials, no
GDAL, and no network — essential for unit tests, CI, and reproducing the
figures in MQFS Working Paper No. 3.

The generative model mirrors the one the pricing engine assumes, so a
calibration round-trip (generate -> calibrate -> compare) is a valid sanity
check on the estimator:

    NDVI(t) = seasonal_mean(doy) + X(t)          (+ optional drought jump)
    X(t)    : AR(1) / discrete Ornstein-Uhlenbeck anomaly
    seasonal_mean(doy): double-logistic-ish annual greenup for winter cereal

Durum wheat in Sicily: sowing ~Nov, greenup through winter, peak ~April,
senescence by June. Drought years depress the spring peak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _seasonal_mean(doy: np.ndarray) -> np.ndarray:
    """Annual NDVI climatology for a Mediterranean winter cereal, in [~0.15, ~0.75].

    Peak near day-of-year 105 (~mid-April), trough in summer (bare/stubble soil).
    """
    phase = 2 * np.pi * (doy - 105) / 365.0
    base = 0.42 + 0.30 * np.cos(phase)          # primary annual harmonic
    base += 0.04 * np.cos(2 * phase)            # asymmetry (fast greenup, slow senescence)
    return np.clip(base, 0.12, 0.80)


def generate_zone(
    start: str = "2016-01-01",
    end: str = "2026-05-01",
    cadence_days: int = 10,
    kappa: float = 6.0,        # OU mean-reversion speed (per year)
    sigma: float = 0.12,       # OU anomaly volatility (per sqrt-year)
    drought_years: tuple = (2017, 2020, 2023, 2026),
    drought_depth: float = 0.14,
    cloud_gap_prob: float = 0.12,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate one zone's dekadal NDVI/NDMI series with anomalies and droughts."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq=f"{cadence_days}D")
    doy = dates.dayofyear.to_numpy()
    years = dates.year.to_numpy()
    dt = cadence_days / 365.0

    a = np.exp(-kappa * dt)
    b = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))

    x = 0.0
    anomaly = np.empty(len(dates))
    for i in range(len(dates)):
        x = a * x + b * rng.standard_normal()
        anomaly[i] = x

    # Drought = persistent negative spring anomaly in selected years.
    drought = np.zeros(len(dates))
    spring = (doy >= 60) & (doy <= 150)
    for yr in drought_years:
        sel = spring & (years == yr)
        drought[sel] -= drought_depth * rng.uniform(0.7, 1.3)

    ndvi = _seasonal_mean(doy) + anomaly + drought
    ndvi = np.clip(ndvi + 0.01 * rng.standard_normal(len(dates)), -0.2, 0.95)

    # NDMI (moisture) is correlated with NDVI but lower-amplitude.
    ndmi = np.clip(0.55 * ndvi + 0.05 + 0.03 * rng.standard_normal(len(dates)), -0.3, 0.6)

    df = pd.DataFrame({"date": dates, "NDVI": ndvi, "NDMI": ndmi})

    # Inject cloud gaps (NaN) the way real S2 dekadal composites have them.
    gap = rng.random(len(df)) < cloud_gap_prob
    df.loc[gap, ["NDVI", "NDMI"]] = np.nan
    return df


def generate_panel(zone_ids, **kwargs) -> dict:
    """Generate a {zone_id: DataFrame} panel; each zone gets a distinct seed."""
    return {zid: generate_zone(seed=7 + i, **kwargs) for i, zid in enumerate(zone_ids)}


if __name__ == "__main__":
    df = generate_zone()
    print(df.describe())
    print("rows:", len(df), "| NaN frac:", df["NDVI"].isna().mean().round(3))
