"""
oracle.indices
==============
Vectorised spectral-index kernels. Pure functions, no I/O, no GEE/raster
dependency, so they unit-test in milliseconds and run on NumPy arrays,
``xarray.DataArray`` or Google Earth Engine ``ee.Image`` alike (any object
that supports element-wise ``+ - * /``).

Index definitions (Sentinel-2 band convention)
-----------------------------------------------
NDVI  (vegetation vigour)            = (NIR  - RED ) / (NIR  + RED )  = (B8  - B4 ) / (B8  + B4 )
NDWI  (McFeeters, open water)        = (GREEN- NIR ) / (GREEN+ NIR )  = (B3  - B8 ) / (B3  + B8 )
NDMI  (Gao, canopy water content)    = (NIR  - SWIR) / (NIR  + SWIR)  = (B8  - B11) / (B8  + B11)

Quant note
----------
The user brief asks for "NDWI". For *crop drought stress* the McFeeters NDWI
(green/NIR) actually tracks **open water**, not vegetation moisture. The index
that is diagnostic of canopy water deficit is the Gao formulation (NIR/SWIR),
usually labelled **NDMI/NDWI_Gao**. We compute both and use NDMI as the moisture
stressor in the signal layer, while NDVI remains the primary biomass proxy.
This distinction matters for basis risk and is made explicit in the paper.
"""
from __future__ import annotations

# epsilon guards division where (A + B) -> 0 over masked / no-data pixels.
_EPS = 1e-6


def _normalized_difference(a, b):
    """(a - b) / (a + b) with a small epsilon to avoid 0/0 on masked pixels."""
    return (a - b) / (a + b + _EPS)


def ndvi(nir, red):
    """Normalized Difference Vegetation Index — canopy greenness / biomass proxy."""
    return _normalized_difference(nir, red)


def ndwi_mcfeeters(green, nir):
    """McFeeters (1996) NDWI — delineates open water bodies. NOT canopy moisture."""
    return _normalized_difference(green, nir)


def ndmi(nir, swir1):
    """Gao (1996) NDMI / NDWI_Gao — vegetation liquid-water content; drought stress."""
    return _normalized_difference(nir, swir1)


# Backwards-compatible alias: many users mean "canopy moisture" when they say NDWI.
def ndwi(nir, swir1):
    """Alias for :func:`ndmi`: the moisture-stress index used by the quant layer."""
    return ndmi(nir, swir1)


def evi(nir, red, blue, G=2.5, C1=6.0, C2=7.5, L=1.0):
    """Enhanced Vegetation Index — saturates less than NDVI over dense canopy.

    Useful as a robustness cross-check for high-biomass citrus/vineyard zones.
    """
    return G * (nir - red) / (nir + C1 * red - C2 * blue + L + _EPS)


def all_indices(bands: dict) -> dict:
    """Compute every supported index from a dict of band arrays.

    Parameters
    ----------
    bands : dict
        Keys among {'blue','green','red','nir','swir1'} mapping to arrays.

    Returns
    -------
    dict of index_name -> array
    """
    out = {}
    if {"nir", "red"} <= bands.keys():
        out["NDVI"] = ndvi(bands["nir"], bands["red"])
    if {"green", "nir"} <= bands.keys():
        out["NDWI"] = ndwi_mcfeeters(bands["green"], bands["nir"])
    if {"nir", "swir1"} <= bands.keys():
        out["NDMI"] = ndmi(bands["nir"], bands["swir1"])
    if {"nir", "red", "blue"} <= bands.keys():
        out["EVI"] = evi(bands["nir"], bands["red"], bands["blue"])
    return out
