"""
Shared pytest fixtures and path setup.

The package uses absolute imports rooted at ``src`` (``from pricing_engine...``,
``from quant_layer...``, ``from oracle...``), so we put ``src`` on ``sys.path``
once here rather than installing the package for the test run.
"""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@pytest.fixture(scope="session")
def synthetic_zone():
    """A single zone's dekadal NDVI/NDMI panel (deterministic seed)."""
    from oracle.synthetic import generate_zone
    return generate_zone(seed=7)


@pytest.fixture(scope="session")
def features(synthetic_zone):
    """Engineered feature frame for the synthetic zone."""
    from quant_layer.feature_engineering import build_features
    return build_features(synthetic_zone, ndvi_col="NDVI")


@pytest.fixture(scope="session")
def seasonal_climatology(features):
    """Climatology evaluated on the dekadal risk window (DOY 75..151)."""
    from pricing_engine.monte_carlo import seasonal_grid
    clim = features.attrs["climatology"]
    return seasonal_grid(clim, 75, 151, step_days=10)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(12345)
