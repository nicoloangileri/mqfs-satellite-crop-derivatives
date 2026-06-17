"""
quant_layer.anomaly_detection
==============================
Isolate the *yield-shock* signal: the event we are actually selling insurance
against. A shock is a statistically significant, persistent collapse of crop
health below its seasonal expectation.

Two complementary triggers (config-driven):
  * z-score trigger : standardized anomaly below ``zscore_trigger`` (default -1.5)
  * VCI trigger     : Vegetation Condition Index below ``vci_trigger`` (default 0.35)

We also collapse consecutive flagged dekads into discrete **drought episodes**
with a depth (integrated shortfall) and duration — the empirical objects whose
distribution the Monte Carlo engine must reproduce.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List
import numpy as np
import pandas as pd


@dataclass
class DroughtEpisode:
    start: pd.Timestamp
    end: pd.Timestamp
    duration_dekads: int
    min_zscore: float
    min_vci: float
    integrated_shortfall: float   # sum of negative anomaly over the episode (severity)
    peak: pd.Timestamp = None     # dekad of maximum stress (most negative anomaly_z)


def flag_shocks(features: pd.DataFrame, zscore_trigger: float = -1.5,
                vci_trigger: float = 0.35) -> pd.Series:
    """Boolean Series: dekads in crop-stress state (either trigger fired)."""
    z_hit = features["anomaly_z"] < zscore_trigger
    vci_hit = features["vci"] < vci_trigger
    return (z_hit | vci_hit).rename("shock")


def detect_episodes(features: pd.DataFrame, shock: pd.Series | None = None,
                    min_duration: int = 1, max_gap: int = 0,
                    **trigger_kw) -> List[DroughtEpisode]:
    """Group consecutive shock dekads into drought episodes.

    ``max_gap`` tolerates short recoveries: runs separated by at most
    ``max_gap`` non-shock dekads are merged into a single episode (real S2
    composites are noisy, and a one-dekad rebound rarely ends a drought).
    ``min_duration`` drops episodes shorter than the given number of dekads.
    """
    if shock is None:
        shock = flag_shocks(features, **trigger_kw)
    flags = shock.to_numpy()
    idx = features.index

    episodes: List[DroughtEpisode] = []
    in_run, start_idx, last_hit, gap = False, None, None, 0

    for i, flag in enumerate(flags):
        if flag:
            if not in_run:
                in_run, start_idx = True, i
            last_hit, gap = i, 0
        elif in_run:
            gap += 1
            if gap > max_gap:                       # sustained recovery -> close
                episodes.append(_make_episode(features, idx, start_idx, last_hit))
                in_run = False
    if in_run:
        episodes.append(_make_episode(features, idx, start_idx, last_hit))

    return [e for e in episodes if e.duration_dekads >= min_duration]


def _make_episode(features, idx, i0, i1) -> DroughtEpisode:
    seg = features.iloc[i0:i1 + 1]
    neg = seg["anomaly"].clip(upper=0.0)
    return DroughtEpisode(
        start=idx[i0], end=idx[i1],
        duration_dekads=i1 - i0 + 1,
        min_zscore=float(seg["anomaly_z"].min()),
        min_vci=float(seg["vci"].min()),
        integrated_shortfall=float(neg.sum()),
        peak=seg["anomaly_z"].idxmin(),
    )


def episodes_to_frame(episodes: List[DroughtEpisode]) -> pd.DataFrame:
    return pd.DataFrame([asdict(e) for e in episodes])


def seasonal_index(features: pd.DataFrame, start_doy: int, end_doy: int,
                   col: str = "ndvi_smooth", how: str = "mean") -> pd.Series:
    """Collapse each calendar year to one seasonal index value over the risk window.

    This is the settlement variable of the parametric contract: one number per
    season that the payoff is written on. ``how`` in {mean, integral, min}.
    """
    df = features.copy()
    doy = df.index.dayofyear
    in_win = (doy >= start_doy) & (doy <= end_doy)
    sel = df.loc[in_win, col]
    grp = sel.groupby(sel.index.year)
    if how == "mean":
        out = grp.mean()
    elif how == "integral":          # iNDVI proxy (sum of dekadal values)
        out = grp.sum()
    elif how == "min":
        out = grp.min()
    else:
        raise ValueError(f"unknown index aggregator {how!r}")
    out.index.name = "year"
    return out.rename(f"seasonal_{col}_{how}")
