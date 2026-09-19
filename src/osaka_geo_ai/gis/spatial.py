"""Small, testable spatial operations. All distances/areas require metres."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from osaka_geo_ai.data.validation import validate_geo


def compatible(points, other):
    validate_geo(points, "points", projected=True)
    validate_geo(other, "spatial layer", projected=True)
    if points.crs != other.crs:
        raise ValueError("Spatial layers must use the same projected CRS")


def station_features(points, stations, radius_m=1000):
    compatible(points, stations)
    if radius_m <= 0:
        raise ValueError("Station radius must be positive")
    if not points.geom_type.eq("Point").all() or not stations.geom_type.eq("Point").all():
        raise ValueError("Station distances require representative point geometries")
    tree = cKDTree(np.column_stack([stations.geometry.x, stations.geometry.y]))
    xy = np.column_stack([points.geometry.x, points.geometry.y])
    distance, _ = tree.query(xy, k=1)
    counts = tree.query_ball_point(xy, radius_m, return_length=True)
    return pd.DataFrame(
        {"nearest_station_distance_m": distance, "station_count_1km": counts}, index=points.index
    )


def polygon_features(points, polygons, columns, key):
    compatible(points, polygons)
    try:
        joined = gpd.sjoin(
            points[["geometry"]],
            polygons[[key, *columns, "geometry"]],
            how="left",
            predicate="intersects",
        )
    except Exception as exc:
        raise RuntimeError(f"Population spatial join failed: {exc}") from exc
    # A point exactly on a mesh edge may touch two cells. Pick the smallest mesh ID deterministically.
    joined = joined.sort_values(key, na_position="last").loc[
        lambda x: ~x.index.duplicated(keep="first")
    ]
    return joined[[key, *columns]].reindex(points.index)


def landuse_features(points, polygons, classes, radius_m):
    compatible(points, polygons)
    if not polygons.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Land-use features require polygon meshes")
    buffers = points.geometry.buffer(radius_m)
    left, right = polygons.sindex.query(buffers, predicate="intersects")
    areas = shapely.area(
        shapely.intersection(buffers.to_numpy()[left], polygons.geometry.to_numpy()[right])
    )
    pairs = pd.DataFrame(
        {"point": left, "class": polygons.landuse_code.to_numpy()[right], "area": areas}
    )
    denominator = buffers.area.to_numpy()
    totals = pairs.groupby("point").area.sum().reindex(range(len(points)), fill_value=0).to_numpy()
    coverage = totals / denominator
    if (coverage > 1.001).any():
        raise ValueError("Land-use polygons overlap; composition ratios would be double counted")
    result = pd.DataFrame(index=points.index)
    result["landuse_coverage_ratio"] = coverage
    for name, codes in classes.items():
        values = (
            pairs[pairs["class"].isin(codes)]
            .groupby("point")
            .area.sum()
            .reindex(range(len(points)), fill_value=0)
            .to_numpy()
        )
        result[f"{name}_ratio"] = np.where(coverage >= 0.95, values / denominator, np.nan)
    return result


def flood_features(points, flood, coverage):
    compatible(points, flood)
    compatible(points, coverage)
    covered = points.geometry.intersects(coverage.geometry.union_all())
    joined = gpd.sjoin(
        points[["geometry"]],
        flood[["flood_depth_m", "geometry"]],
        how="left",
        predicate="intersects",
    )
    depth = joined.groupby(level=0).flood_depth_m.max().reindex(points.index)
    result = pd.DataFrame(index=points.index)
    result["flood_depth_m"] = depth.fillna(0).where(covered)
    result["flood_risk"] = depth.notna().astype(float).where(covered)
    result["flood_coverage_known"] = covered
    return result
