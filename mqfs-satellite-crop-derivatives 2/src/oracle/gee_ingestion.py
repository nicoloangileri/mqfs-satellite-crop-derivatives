"""
oracle.gee_ingestion
=====================
Server-side ingestion of a multi-year NDVI/NDMI time series for Sicilian
cropland using Google Earth Engine (GEE).

Why server-side
---------------
A 10-year, 20 m, multi-band stack over Sicily is tens of terabytes. Downloading
it to compute indices locally is the amateur path and will not scale. The
institutional pattern is to push the *reduction* to Google's cluster: mask
clouds, compute the index, mask to cropland, and spatially reduce to a single
mean per dekad per zone **before** anything crosses the network. What returns is
a few kilobytes of time series — the "alpha", already distilled.

Requirements
------------
    pip install earthengine-api
    earthengine authenticate          # one-time OAuth (do NOT hard-code keys)

This module imports lazily so the rest of the repo (quant layer + pricing) runs
with zero geospatial dependencies for CI / offline testing.
"""
from __future__ import annotations

from typing import Dict, List
import pandas as pd


def _require_ee():
    try:
        import ee  # type: ignore
        return ee
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "earthengine-api not installed. `pip install earthengine-api` and "
            "run `earthengine authenticate`. The quant/pricing layers do not "
            "need this dependency."
        ) from exc


def init_ee(project: str | None = None) -> None:
    """Initialise the Earth Engine session (OAuth must already be configured)."""
    ee = _require_ee()
    ee.Initialize(project=project)


# --------------------------------------------------------------------------- #
# Cloud / shadow masking via the Scene Classification Layer (SCL)
# --------------------------------------------------------------------------- #
# SCL classes to discard: 3 cloud-shadow, 8 cloud medium-prob, 9 cloud high-prob,
# 10 thin cirrus, 11 snow. Keep vegetation (4), bare (5), water (6) etc.
_SCL_REJECT = [3, 8, 9, 10, 11]


def _mask_clouds(img, scl_band: str = "SCL"):
    ee = _require_ee()
    scl = img.select(scl_band)
    mask = ee.Image.constant(1)
    for c in _SCL_REJECT:
        mask = mask.And(scl.neq(c))
    # Sentinel-2 SR is scaled by 1e4; rescale reflectance to [0, 1].
    return img.updateMask(mask).divide(10000).copyProperties(img, ["system:time_start"])


def _add_indices(img, b: Dict[str, str]):
    """Attach NDVI and NDMI bands (Gao moisture) computed server-side."""
    ndvi = img.normalizedDifference([b["nir"], b["red"]]).rename("NDVI")
    ndmi = img.normalizedDifference([b["nir"], b["swir1"]]).rename("NDMI")
    return img.addBands([ndvi, ndmi])


def build_collection(cfg: dict):
    """Construct the masked, index-augmented Sentinel-2 SR collection."""
    ee = _require_ee()
    s = cfg["sensor"]
    a = cfg["aoi"]
    region = ee.Geometry.Rectangle(a["bbox"])
    b = s["bands"]

    coll = (
        ee.ImageCollection(s["mission"])
        .filterBounds(region)
        .filterDate(s["start_date"], s["end_date"])
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", s["max_cloud_pct"]))
        .map(lambda im: _mask_clouds(im, b["scl"]))
        .map(lambda im: _add_indices(im, b))
    )
    return coll, region


def _cropland_mask(cfg: dict):
    """ESA WorldCover cropland (class 40) as a multiply-able 0/1 mask."""
    ee = _require_ee()
    lc = cfg["landcover"]
    wc = ee.ImageCollection(lc["product"]).first().select("Map")
    return wc.eq(lc["cropland_class"])


def _dekad_starts(start_date: str, end_date: str):
    """Dekadal (10-day) composite anchors: days 1, 11, 21 of each month."""
    ee = _require_ee()
    months = ee.List.sequence(0, ee.Date(end_date).difference(ee.Date(start_date), "month").subtract(1))

    def per_month(m):
        base = ee.Date(start_date).advance(m, "month")
        return ee.List([base, base.advance(10, "day"), base.advance(20, "day")])

    return months.map(per_month).flatten()


def extract_zone_timeseries(cfg: dict, zone_bbox: List[float]) -> pd.DataFrame:
    """Return a dekadal NDVI/NDMI mean time series for one zone as a DataFrame.

    The heavy lifting (mask -> index -> reduceRegion) happens on GEE; only the
    distilled per-dekad means are pulled back via ``getInfo``.
    """
    ee = _require_ee()
    coll, _ = build_collection(cfg)
    crop = _cropland_mask(cfg)
    coll = coll.map(lambda im: im.updateMask(crop))
    geom = ee.Geometry.Rectangle(zone_bbox)
    scale = cfg["sensor"]["scale_m"]

    anchors = _dekad_starts(cfg["sensor"]["start_date"], cfg["sensor"]["end_date"])

    def reduce_dekad(d0):
        d0 = ee.Date(d0)
        window = coll.filterDate(d0, d0.advance(10, "day"))
        composite = window.select(["NDVI", "NDMI"]).median()
        stats = composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale,
            maxPixels=1e10,
            bestEffort=True,
        )
        return ee.Feature(None, {
            "date": d0.format("YYYY-MM-dd"),
            "NDVI": stats.get("NDVI"),
            "NDMI": stats.get("NDMI"),
        })

    fc = ee.FeatureCollection(anchors.map(reduce_dekad))
    rows = fc.getInfo()["features"]
    df = pd.DataFrame([r["properties"] for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna(subset=["NDVI"]).sort_values("date").reset_index(drop=True)


def extract_all_zones(cfg: dict) -> Dict[str, pd.DataFrame]:
    """Ingest every configured zone -> {zone_id: DataFrame}."""
    out = {}
    for z in cfg["aoi"]["zones"]:
        out[z["id"]] = extract_zone_timeseries(cfg, z["bbox"])
    return out


if __name__ == "__main__":  # pragma: no cover
    import yaml, pathlib
    cfg = yaml.safe_load(open(pathlib.Path(__file__).parents[2] / "config" / "config.yaml"))
    init_ee(cfg["project"].get("gee_project"))
    df = extract_zone_timeseries(cfg, cfg["aoi"]["zones"][0]["bbox"])
    print(df.tail())
