"""Fail explicitly on missing CRS, malformed geometry and incomplete schemas."""

import numpy as np
from pyproj import CRS


def require_columns(frame, columns, name="dataset"):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: required columns missing: {sorted(missing)}")


def validate_geo(frame, name="dataset", projected=False):
    if frame.crs is None:
        raise ValueError(f"{name}: CRS is missing; do not infer it from coordinate values")
    crs = CRS(frame.crs)
    if projected and (not crs.is_projected or any(a.unit_name != "metre" for a in crs.axis_info)):
        raise ValueError(f"{name}: distance/area requires a projected CRS in metres")
    if frame.empty:
        raise ValueError(f"{name}: no features")
    if (
        frame.geometry.isna().any()
        or frame.geometry.is_empty.any()
        or not frame.geometry.is_valid.all()
    ):
        raise ValueError(f"{name}: null, empty or invalid geometries")
    if not np.isfinite(frame.total_bounds).all():
        raise ValueError(f"{name}: non-finite coordinates")


def validate_unique(frame, keys, name="dataset"):
    if frame.duplicated(keys).any():
        raise ValueError(f"{name}: duplicate keys {keys}")
