"""Normalize annual releases independently; retain no future-year attributes."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from osaka_geo_ai.config import artifact, directory
from osaka_geo_ai.data.loader import read_archive, read_geo
from osaka_geo_ai.data.validation import require_columns, validate_geo, validate_unique
from osaka_geo_ai.io import write_json

LOG = logging.getLogger(__name__)


def code(series, width):
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(width)


def normalize_land(frame, item, prefecture_code, crs):
    mapping = item["columns"]
    require_columns(frame, mapping.values(), f"land price {item['year']}")
    frame = (
        frame[list(mapping.values()) + ["geometry"]]
        .rename(columns={v: k for k, v in mapping.items()})
        .copy()
    )
    frame["municipality_code"] = code(frame.municipality_code, 5)
    frame = frame[frame.municipality_code.str.startswith(prefecture_code)].copy()
    for name in ("use_code", "sequence", "previous_use_code", "previous_sequence"):
        frame[name] = code(frame[name], 3)
    if "previous_municipality_code" not in frame:
        # Legacy release has no previous municipality. Do not guess municipal reorganizations.
        frame["previous_municipality_code"] = frame.municipality_code
    frame["previous_municipality_code"] = code(frame.previous_municipality_code, 5)
    frame["site_id"] = frame.municipality_code + "-" + frame.use_code + "-" + frame.sequence
    frame["previous_site_id"] = (
        frame.previous_municipality_code
        + "-"
        + frame.previous_use_code
        + "-"
        + frame.previous_sequence
    )
    for name in ("year", "land_price", "area_m2", "selection"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if not frame.year.eq(item["year"]).all():
        raise ValueError(f"Release year mismatch: {item['filename']}")
    if not frame.land_price.gt(0).all():
        raise ValueError("Land prices must be strictly positive")
    frame["zoning_code"] = frame.zoning_code.fillna("unknown").astype(str)
    frame["row_id"] = frame.year.astype(str) + ":" + frame.site_id
    validate_unique(frame, ["row_id"], "land price")
    validate_geo(frame, "Osaka land price")
    if not frame.geom_type.eq("Point").all():
        raise ValueError("Land price geometry must be Point")
    return frame.to_crs(crs)


def deduplicate_stations(frame, merge_m):
    """Merge same-name station sections whose projected centroids are within merge_m."""
    validate_geo(frame, "stations", projected=True)
    frame = frame[["N02_005", "geometry"]].rename(columns={"N02_005": "station_name"}).copy()
    frame.geometry = frame.geometry.centroid
    rows = []
    for name, group in frame.groupby("station_name", sort=True):
        # Connected components avoid rounding-grid artefacts; typically a handful of sections.
        remaining = set(range(len(group)))
        geometries = list(group.geometry)
        while remaining:
            component = {min(remaining)}
            frontier = list(component)
            remaining -= component
            while frontier:
                current = frontier.pop()
                neighbours = {
                    i for i in remaining if geometries[current].distance(geometries[i]) <= merge_m
                }
                remaining -= neighbours
                component |= neighbours
                frontier.extend(neighbours)
            centroid = (
                gpd.GeoSeries([geometries[i] for i in sorted(component)], crs=frame.crs)
                .union_all()
                .centroid
            )
            rows.append({"station_name": name, "geometry": centroid})
    return gpd.GeoDataFrame(rows, crs=frame.crs)


def prepare(config):
    datasets = config["datasets"]
    raw, interim = directory(config, "raw"), directory(config, "interim")
    interim.mkdir(parents=True, exist_ok=True)
    crs = config["analysis"]["analysis_crs"]
    prefecture = config["project"]["prefecture_code"]
    frames, sources = [], []
    for item in datasets["land_price"]["files"]:
        LOG.info("Loading Osaka land prices %s", item["year"])
        frame = read_archive(
            raw / "land_price" / item["filename"],
            item["member"],
            columns=list(item["columns"].values()),
        )
        sources.append(
            {
                "dataset": "land_price",
                "year": item["year"],
                "input_crs": str(frame.crs),
                "rows": len(frame),
            }
        )
        frames.append(normalize_land(frame, item, prefecture, crs))
    land = (
        gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=crs)
        .sort_values(["year", "site_id"])
        .reset_index(drop=True)
    )
    land.to_parquet(interim / "land_price.parquet", index=False)
    context = box(*land.total_bounds).buffer(config["analysis"]["context_buffer_m"])
    context_geo = gpd.GeoSeries([context], crs=crs).to_crs("EPSG:4326").total_bounds

    pop_spec = datasets["population"]
    item = pop_spec["files"][0]
    pop = read_archive(
        raw / "population" / item["filename"],
        item["member"],
        columns=list(pop_spec["columns"].values()),
    )
    sources.append({"dataset": "population", "input_crs": str(pop.crs)})
    require_columns(pop, pop_spec["columns"].values(), "population")
    pop = pop.rename(columns={v: k for k, v in pop_spec["columns"].items()}).to_crs(crs)
    pop["municipality_code"] = code(pop.municipality_code, 5)
    pop = pop[pop.municipality_code.str.startswith(prefecture)].copy()
    pop["mesh_id"] = code(pop.mesh_id, 9)
    validate_unique(pop, ["mesh_id"], "population")
    for name in pop_spec["columns"]:
        if name.startswith(("population_", "elderly_")):
            pop[name] = pd.to_numeric(pop[name], errors="raise").where(lambda x: x >= 0)
    pop["population_projected_change_rate_2010_2020"] = (
        pop.population_projected_2020 / pop.population_500m_2010.replace(0, np.nan) - 1
    )
    pop["elderly_ratio_projected_2020"] = (
        pop.elderly_projected_2020 / pop.population_projected_2020.replace(0, np.nan)
    )
    pop["elderly_ratio_projected_2020"] = pop.elderly_ratio_projected_2020.where(
        pop.elderly_ratio_projected_2020.between(0, 1)
    )
    pop.to_parquet(interim / "population.parquet", index=False)

    item = datasets["railway"]["files"][0]
    for kind, member in [("railway", item["railway_member"]), ("stations", item["station_member"])]:
        frame = read_archive(
            raw / "railway" / item["filename"], member, bbox=tuple(context_geo)
        ).to_crs(crs)
        frame = frame[frame.intersects(context)].copy()
        if kind == "stations":
            frame = deduplicate_stations(frame, config["analysis"]["station_merge_m"])
        validate_geo(frame, kind, projected=True)
        frame.to_parquet(interim / f"{kind}.parquet", index=False)

    spec = datasets["landuse"]
    if spec.get("enabled"):
        parts = []
        for item in spec["files"]:
            LOG.info("Loading land-use tile %s", item["filename"])
            frame = read_archive(
                raw / "landuse" / item["filename"],
                item["member"],
                bbox=tuple(context_geo),
                allow_empty=True,
            )
            if frame.empty:
                continue
            require_columns(frame, [spec["mesh_column"], spec["class_column"]], "landuse")
            parts.append(frame[[spec["mesh_column"], spec["class_column"], "geometry"]].to_crs(crs))
        landuse = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs).drop_duplicates(
            spec["mesh_column"]
        )
        landuse = landuse[landuse.intersects(context)].copy()
        landuse = landuse.rename(
            columns={spec["mesh_column"]: "mesh_id", spec["class_column"]: "landuse_code"}
        )
        landuse["landuse_code"] = code(landuse.landuse_code, 4)
        validate_geo(landuse, "landuse")
        landuse.to_parquet(interim / "landuse.parquet", index=False)

    spec = datasets["boundary"]
    item = spec["files"][0]
    boundary = read_archive(raw / "boundary" / item["filename"], item["member"]).to_crs(crs)
    require_columns(boundary, ["N03_007"], "boundary")
    boundary = boundary[code(boundary.N03_007, 5).str.startswith(prefecture)]
    boundary = boundary[["N03_007", "geometry"]].dissolve("N03_007").reset_index()
    boundary.to_parquet(interim / "boundary.parquet", index=False)

    flood = datasets["flood"]
    if flood.get("enabled"):
        for key, filename in [
            ("manual_path", "flood.parquet"),
            ("coverage_path", "flood_coverage.parquet"),
        ]:
            frame = read_geo(Path(config["_root"]) / flood[key]).to_crs(crs)
            if key == "manual_path":
                require_columns(frame, [flood["depth_column"]], "flood")
                frame = frame.rename(columns={flood["depth_column"]: "flood_depth_m"})
                if not frame.flood_depth_m.ge(0).all():
                    raise ValueError(
                        "Flood depth must be nonnegative metres; class codes are not depths"
                    )
            frame.to_parquet(interim / filename, index=False)
    summary = {
        "schema_version": "1.0",
        "region": config["project"]["region"],
        "land_price_rows_by_year": land.groupby("year").size().to_dict(),
        "population_meshes": len(pop),
        "population_observed_year": 2010,
        "population_projection_vintage": pop_spec["year"],
        "population_projection_years": [2020, 2025],
        "analysis_crs": crs,
        "input_sources": sources,
        "flood_enabled": flood["enabled"],
        "landuse_enabled": datasets["landuse"].get("enabled", True),
    }
    write_json(artifact(config, "dataset_summary.json"), summary)
    return interim / "land_price.parquet"
