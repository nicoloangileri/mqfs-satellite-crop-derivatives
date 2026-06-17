"""
oracle.sentinel_processor
=========================
Local / on-prem processing path for Sentinel-2 L2A tiles obtained from the
Copernicus Data Space (``sentinelhub`` or the openEO API) when GEE is not
available or when raw pixels must stay in-house for compliance.

Memory discipline (the user's explicit requirement)
---------------------------------------------------
Satellite arrays blow up RAM if read eagerly. The rules enforced here:

1. **Lazy, chunked reads** via ``rioxarray.open_rasterio(..., chunks=...)`` ->
   Dask arrays. Nothing is materialised until ``.compute()``.
2. **Windowed reduction**: indices and zonal means are expressed as Dask graphs
   and reduced *before* calling compute, so peak memory ~ one chunk, not one
   scene.
3. **Context-managed handles**: every dataset is opened in a ``with`` block; GDAL
   file descriptors are released deterministically (a classic leak source).
4. **float32 throughout**: reflectance never needs float64; halves the footprint.

Lazy imports keep the package importable without GDAL installed.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from .indices import ndvi as _ndvi, ndmi as _ndmi


def _require_geo():
    try:
        import rioxarray  # noqa: F401  (registers the .rio accessor)
        import xarray as xr
        import geopandas as gpd  # noqa: F401
        return xr
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Local raster path needs: rioxarray xarray geopandas dask rasterio. "
            "`pip install rioxarray geopandas dask[array] rasterio`. "
            "Use the GEE path or the synthetic generator if these are unavailable."
        ) from exc


# SCL classes treated as invalid (see gee_ingestion for the same convention).
_SCL_REJECT = (3, 8, 9, 10, 11)
_CHUNK = {"x": 1024, "y": 1024}


def load_scene(band_paths: Dict[str, str], scl_path: str):
    """Open one Sentinel-2 scene lazily as a chunked xarray Dataset.

    Parameters
    ----------
    band_paths : dict
        {'red': '...B4.tif', 'nir': '...B8.tif', 'swir1': '...B11.tif', ...}.
        SWIR is 20 m; it is reprojected to the 10 m grid on read.
    scl_path : str
        Path to the Scene Classification Layer GeoTIFF.
    """
    xr = _require_geo()
    import rioxarray  # noqa: F401

    arrays = {}
    ref = None
    for name, path in band_paths.items():
        da = rioxarray.open_rasterio(path, chunks=_CHUNK, masked=True).squeeze("band", drop=True)
        da = da.astype("float32") / 10000.0   # SR scale factor -> reflectance
        if ref is None:
            ref = da
        else:
            da = da.rio.reproject_match(ref)   # align grids (handles 10 m vs 20 m)
        arrays[name] = da

    scl = rioxarray.open_rasterio(scl_path, chunks=_CHUNK).squeeze("band", drop=True)
    scl = scl.rio.reproject_match(ref)
    arrays["SCL"] = scl
    return xr.Dataset(arrays)


def mask_and_index(ds):
    """Apply the SCL cloud mask and attach NDVI / NDMI as lazy bands."""
    valid = ~ds["SCL"].isin(_SCL_REJECT)
    nir = ds["nir"].where(valid)
    red = ds["red"].where(valid)
    swir1 = ds["swir1"].where(valid)
    ds = ds.assign(NDVI=_ndvi(nir, red).astype("float32"),
                   NDMI=_ndmi(nir, swir1).astype("float32"))
    return ds


def zonal_mean(ds, zones_gdf, index: str = "NDVI") -> pd.Series:
    """Mean of ``index`` per geometry in ``zones_gdf`` (cropland-clipped upstream).

    Uses ``exactextract`` if available (sub-pixel exact weights), else falls back
    to a clip + nan-mean. The reduction is a Dask graph; only the scalar means
    are computed.
    """
    _require_geo()
    try:
        from exactextract import exact_extract  # type: ignore
        res = exact_extract(ds[index], zones_gdf, ["mean"], output="pandas")
        return res["mean"]
    except ImportError:
        means = {}
        for idx, row in zones_gdf.iterrows():
            clip = ds[index].rio.clip([row.geometry], zones_gdf.crs, drop=True)
            means[idx] = float(clip.mean(skipna=True).compute())
        return pd.Series(means, name=f"{index}_mean")


def process_archive_to_panel(scene_index: pd.DataFrame, zones_gdf,
                             cropland_gdf=None) -> pd.DataFrame:
    """Walk a catalogue of dated scenes -> tidy (date, zone, NDVI, NDMI) panel.

    Parameters
    ----------
    scene_index : DataFrame
        Columns: date, plus band path columns (red, nir, swir1, ...), scl.
    zones_gdf : GeoDataFrame
        Pricing zones (one contract each).
    cropland_gdf : GeoDataFrame, optional
        Cropland polygons to intersect zones with before reduction.
    """
    _require_geo()
    if cropland_gdf is not None:
        import geopandas as gpd
        zones_gdf = gpd.overlay(zones_gdf, cropland_gdf, how="intersection")

    records = []
    for _, sc in scene_index.iterrows():
        band_paths = {k: sc[k] for k in ("red", "nir", "swir1") if k in sc}
        with load_scene(band_paths, sc["scl"]) as ds:   # deterministic FD release
            ds = mask_and_index(ds)
            ndvi_z = zonal_mean(ds, zones_gdf, "NDVI")
            ndmi_z = zonal_mean(ds, zones_gdf, "NDMI")
            for zid in zones_gdf.index:
                records.append({"date": sc["date"], "zone": zid,
                                "NDVI": ndvi_z.get(zid), "NDMI": ndmi_z.get(zid)})
    out = pd.DataFrame.from_records(records)
    out["date"] = pd.to_datetime(out["date"])
    return out.dropna(subset=["NDVI"]).sort_values(["zone", "date"]).reset_index(drop=True)
