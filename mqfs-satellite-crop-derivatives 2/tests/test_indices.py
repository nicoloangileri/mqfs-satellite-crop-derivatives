"""Spectral index sanity checks (NDVI / NDWI / NDMI / EVI)."""
import numpy as np

from oracle.indices import ndvi, ndwi_mcfeeters, ndmi, ndwi, evi, all_indices


def test_ndvi_known_value():
    # red=0.1, nir=0.5 -> (0.5-0.1)/(0.5+0.1) = 0.4/0.6
    assert np.isclose(ndvi(0.5, 0.1), 0.4 / 0.6)


def test_ndvi_in_range():
    rng = np.random.default_rng(0)
    nir = rng.uniform(0.0, 1.0, 1000)
    red = rng.uniform(0.0, 1.0, 1000)
    v = ndvi(nir, red)
    assert np.all(v >= -1.0 - 1e-9) and np.all(v <= 1.0 + 1e-9)


def test_dense_vegetation_positive_sparse_negative():
    assert ndvi(0.6, 0.05) > 0.5          # lush canopy
    assert ndvi(0.2, 0.4) < 0.0           # bare/soil dominated


def test_ndmi_alias_is_gao():
    # The public `ndwi` in this package is the moisture index (Gao NIR/SWIR),
    # distinct from McFeeters open-water NDWI.
    assert np.isclose(ndwi(0.4, 0.2), ndmi(0.4, 0.2))
    assert not np.isclose(ndwi_mcfeeters(0.3, 0.4), ndmi(0.4, 0.2))


def test_zero_division_guarded():
    # nir+red == 0 must not blow up
    assert np.isfinite(ndvi(0.0, 0.0))


def test_evi_finite_and_reasonable():
    e = evi(nir=0.5, red=0.1, blue=0.05)
    assert np.isfinite(e) and -1.5 < e < 1.5


def test_all_indices_keys():
    bands = {"red": 0.1, "green": 0.12, "nir": 0.5, "swir1": 0.25, "blue": 0.05}
    out = all_indices(bands)
    for k in ("NDVI", "NDMI"):
        assert k in out and np.isfinite(out[k])
