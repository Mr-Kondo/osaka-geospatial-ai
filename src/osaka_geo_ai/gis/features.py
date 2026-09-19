"""Build point/year features with explicit previous-year links and next-year targets."""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from osaka_geo_ai.config import artifact, directory
from osaka_geo_ai.data.loader import read_geo
from osaka_geo_ai.data.validation import require_columns, validate_geo, validate_unique
from osaka_geo_ai.gis.spatial import (
    flood_features,
    landuse_features,
    polygon_features,
    station_features,
)
from osaka_geo_ai.io import write_json

LOG = logging.getLogger(__name__)


def link_annual_prices(land, max_movement_m):
    """Only official continuous/renumbered sites within the movement bound become labels."""
    validate_geo(land, "land prices", projected=True)
    validate_unique(land, ["row_id"], "land prices")
    frame = land.sort_values(["year", "site_id"]).reset_index(drop=True).copy()
    for col in [
        "previous_land_price",
        "previous_land_price_change",
        "land_price_change_rate",
        "next_year_land_price",
        "next_year_land_price_change_rate",
    ]:
        frame[col] = np.nan
    frame["target_year"] = frame.year + 1
    frame["previous_row_id"] = pd.Series(None, index=frame.index, dtype="object")
    counts = []
    for year in sorted(frame.year.unique())[1:]:
        prior = frame[frame.year.eq(year - 1)].set_index("site_id", drop=False)
        valid = moved = unmatched = 0
        for idx, row in frame[frame.year.eq(year)].iterrows():
            if row.selection not in (1, 2) or row.previous_site_id not in prior.index:
                unmatched += 1
                continue
            previous = prior.loc[row.previous_site_id]
            if (
                row.geometry.distance(previous.geometry) > max_movement_m
                or row.use_code != previous.use_code
            ):
                moved += 1
                continue
            old_idx = frame.index[frame.row_id.eq(previous.row_id)][0]
            change = float(row.land_price / previous.land_price - 1)
            frame.loc[
                idx,
                [
                    "previous_land_price",
                    "previous_land_price_change",
                    "land_price_change_rate",
                    "previous_row_id",
                ],
            ] = [previous.land_price, previous.land_price_change_rate, change, previous.row_id]
            frame.loc[old_idx, ["next_year_land_price", "next_year_land_price_change_rate"]] = [
                row.land_price,
                change,
            ]
            valid += 1
        counts.append(
            {
                "target_year": int(year),
                "linked": valid,
                "excluded_movement_or_use": moved,
                "unmatched_or_new": unmatched,
            }
        )
    return frame, counts


def build_features(config):
    interim = directory(config, "interim")
    land, link_summary = link_annual_prices(
        read_geo(interim / "land_price.parquet"), config["analysis"]["max_site_movement_m"]
    )
    # Geometry deduplication avoids recomputing static context across seven annual releases.
    land["geometry_key"] = land.geometry.to_wkb(hex=True)
    points = (
        land[["geometry_key", "geometry"]].drop_duplicates("geometry_key").reset_index(drop=True)
    )
    LOG.info("Building spatial context for %d distinct positions", len(points))
    stations = station_features(
        points, read_geo(interim / "stations.parquet"), config["analysis"]["station_radius_m"]
    )
    pop_columns = [
        "population_500m_2010",
        "population_projected_2020",
        "population_projected_2025",
        "population_projected_change_rate_2010_2020",
        "elderly_ratio_projected_2020",
    ]
    population = polygon_features(
        points, read_geo(interim / "population.parquet"), pop_columns, "mesh_id"
    )
    parts = [points[["geometry_key"]], stations, population]
    if config["datasets"]["landuse"].get("enabled"):
        LOG.info("Computing area-weighted land-use fractions")
        parts.append(
            landuse_features(
                points,
                read_geo(interim / "landuse.parquet"),
                config["datasets"]["landuse"]["classes"],
                config["analysis"]["landuse_radius_m"],
            )
        )
    else:
        parts.append(
            pd.DataFrame(
                {f"{key}_ratio": np.nan for key in config["datasets"]["landuse"]["classes"]},
                index=points.index,
            )
        )
    if config["datasets"]["flood"].get("enabled"):
        parts.append(
            flood_features(
                points,
                read_geo(interim / "flood.parquet"),
                read_geo(interim / "flood_coverage.parquet"),
            )
        )
    context = pd.concat(parts, axis=1)
    land = land.merge(context, on="geometry_key", how="left", validate="many_to_one").drop(
        columns="geometry_key"
    )
    land = gpd.GeoDataFrame(land, crs=config["analysis"]["analysis_crs"])
    output = land.to_crs(config["analysis"]["output_crs"])
    output["longitude"] = output.geometry.x
    output["latitude"] = output.geometry.y
    # Vintage gate: never fill an earlier origin with a later contextual release.
    gates = {
        "population": pop_columns,
        "railway": list(stations.columns),
        "landuse": [f"{key}_ratio" for key in config["datasets"]["landuse"]["classes"]],
        "flood": ["flood_depth_m", "flood_risk"],
    }
    for dataset, columns in gates.items():
        spec = config["datasets"][dataset]
        if spec.get("enabled"):
            output.loc[
                output.year < spec["available_from_year"], [c for c in columns if c in output]
            ] = np.nan
    require_columns(
        output, config["model"]["numeric_features"] + config["model"]["categorical_features"]
    )
    processed = directory(config, "processed")
    processed.mkdir(parents=True, exist_ok=True)
    output.to_parquet(processed / "features.parquet", index=False)
    write_json(
        artifact(config, "feature_summary.json"),
        {
            "schema_version": "1.0",
            "rows": len(output),
            "annual_links": link_summary,
            "missing_fraction": output.drop(columns="geometry").isna().mean().to_dict(),
            "target_unit": "fraction (0.01 = 1 percentage point)",
            "landuse_denominator": "500m buffer area; null when coverage < 95%",
            "vintage_gates": {k: config["datasets"][k].get("available_from_year") for k in gates},
        },
    )
    return processed / "features.parquet"
