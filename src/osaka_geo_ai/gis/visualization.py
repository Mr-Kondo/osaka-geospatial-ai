"""Comparable static spatial figures with units, fixed extent, attribution and legends."""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[3] / ".cache/matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

from osaka_geo_ai.config import artifact, directory
from osaka_geo_ai.data.loader import read_geo
from osaka_geo_ai.gis.export import interactive_map
from osaka_geo_ai.io import sha256, write_json
from osaka_geo_ai.models.predict import validate_prediction_contract

LOG = logging.getLogger(__name__)
ATTRIBUTION = "Source: MLIT National Land Numerical Information | Processed by Osaka Geospatial AI"


def style_axis(ax, boundary, extent):
    ax.set_facecolor("#edf4f7")
    boundary.plot(ax=ax, facecolor="#fafafa", edgecolor="#b8c4cc", linewidth=0.45, zorder=0)
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km)", fontsize=9)
    ax.set_ylabel("Northing (km)", fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1000:.0f}"))
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.annotate(
        "N",
        xy=(0.94, 0.95),
        xytext=(0.94, 0.85),
        xycoords="axes fraction",
        ha="center",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#273646"},
    )
    x = extent[0] + (extent[2] - extent[0]) * 0.08
    y = extent[1] + (extent[3] - extent[1]) * 0.065
    ax.plot([x, x + 10000], [y, y], color="#273646", linewidth=2)
    ax.text(x + 5000, y + 900, "10 km", ha="center", fontsize=8)


def color_scale(values, divergent=False, logarithmic=False, limit=None):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Cannot draw a map without finite values")
    if divergent:
        width = max(float(np.max(np.abs(values))) if limit is None else limit, 0.01)
        return matplotlib.colormaps["RdBu_r"], TwoSlopeNorm(vmin=-width, vcenter=0, vmax=width)
    lower, upper = float(values.min()), float(values.max())
    if upper <= lower:
        upper = lower + 1
    return matplotlib.colormaps["viridis"], LogNorm(
        vmin=max(lower, 1), vmax=upper
    ) if logarithmic else Normalize(vmin=lower, vmax=upper)


def draw_points(ax, points, column, title, boundary, extent, cmap, norm, unit, size):
    style_axis(ax, boundary, extent)
    valid = points[points[column].notna()]
    scatter = ax.scatter(
        valid.geometry.x,
        valid.geometry.y,
        c=valid[column],
        cmap=cmap,
        norm=norm,
        s=size,
        linewidths=0.12,
        edgecolors="white",
        alpha=0.9,
        zorder=3,
    )
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=12)
    bar = ax.figure.colorbar(scatter, ax=ax, fraction=0.042, pad=0.03, extend="neither")
    bar.set_label(unit, fontsize=8)
    bar.ax.tick_params(labelsize=8)


def context(config):
    boundary = read_geo(directory(config, "interim") / "boundary.parquet")
    bounds = boundary.total_bounds
    extent = [bounds[0] - 2000, bounds[1] - 2000, bounds[2] + 2000, bounds[3] + 2000]
    return boundary, extent


def build_basic_maps(config):
    land = read_geo(directory(config, "interim") / "land_price.parquet")
    latest = land[land.year.eq(config["analysis"]["target_year"])]
    boundary, extent = context(config)
    cmap, norm = color_scale(latest.land_price, logarithmic=True)
    title = f"Osaka | Official land price, {config['analysis']['target_year']}"
    figure_dir = artifact(config, "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 9))
    draw_points(
        ax,
        latest,
        "land_price",
        title,
        boundary,
        extent,
        cmap,
        norm,
        "JPY / square metre (log scale)",
        config["maps"]["point_size"],
    )
    fig.text(0.05, 0.025, ATTRIBUTION + "\nCRS: " + config["analysis"]["analysis_crs"], fontsize=7)
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(figure_dir / "land_price_map.png", dpi=config["maps"]["dpi"])
    plt.close(fig)
    interactive_map(
        latest,
        boundary,
        "land_price",
        title,
        artifact(config, "maps/land_price_map.html"),
        cmap,
        norm,
        "JPY/m2",
        config["maps"]["tiles"],
    )
    return figure_dir / "land_price_map.png"


def build_maps(config):
    validate_prediction_contract(config)
    build_basic_maps(config)
    crs = config["analysis"]["analysis_crs"]
    frame = read_geo(directory(config, "processed") / "features.parquet").to_crs(crs)
    latest = frame[frame.year.eq(config["analysis"]["target_year"])].copy()
    predicted = read_geo(artifact(config, "predictions/predictions.parquet")).to_crs(crs)
    latest["change_pp"] = latest.land_price_change_rate * 100
    predicted["prediction_pp"] = predicted.prediction * 100
    predicted["residual_pp"] = predicted.residual * 100
    population = read_geo(directory(config, "interim") / "population.parquet")
    population["projected_change_pct"] = population.population_projected_change_rate_2010_2020 * 100
    railway = read_geo(directory(config, "interim") / "railway.parquet")
    stations = read_geo(directory(config, "interim") / "stations.parquet")
    boundary, extent = context(config)
    year = config["analysis"]["target_year"]
    limit = float(max(latest.change_pp.abs().max(), predicted.prediction_pp.abs().max()))
    specifications = [
        (
            "land_price_change_map",
            latest,
            "change_pp",
            f"Observed land-price change | {year - 1} to {year}",
            "Change (%)",
            limit,
        ),
        (
            "prediction_map",
            predicted,
            "prediction_pp",
            f"Held-out predicted change | {year - 1} to {year}",
            "Predicted change (%)",
            limit,
        ),
        (
            "residual_map",
            predicted,
            "residual_pp",
            f"Held-out residual | actual minus predicted, {year}",
            "Residual (percentage points)",
            None,
        ),
    ]
    manifest = {
        "schema_version": "1.0",
        "analysis_year": year,
        "analysis_crs": crs,
        "extent": extent,
        "features_sha256": sha256(directory(config, "processed") / "features.parquet"),
        "prediction_metadata_sha256": sha256(
            artifact(config, "predictions/prediction_metadata.json")
        ),
        "maps": [],
    }
    for filename, points, column, title, unit, scale in specifications:
        LOG.info("Rendering %s", filename)
        cmap, norm = color_scale(points[column], divergent=True, limit=scale)
        fig, ax = plt.subplots(figsize=(8, 9))
        draw_points(
            ax,
            points,
            column,
            title,
            boundary,
            extent,
            cmap,
            norm,
            unit,
            config["maps"]["point_size"],
        )
        note = (
            "Positive residual = underprediction; negative = overprediction."
            if filename == "residual_map"
            else "Points are official surveyed sites; they do not cover every property."
        )
        fig.text(0.05, 0.025, f"{note}\n{ATTRIBUTION}", fontsize=7)
        fig.tight_layout(rect=[0, 0.055, 1, 1])
        path = artifact(config, f"figures/{filename}.png")
        fig.savefig(path, dpi=config["maps"]["dpi"])
        plt.close(fig)
        interactive_map(
            points,
            boundary,
            column,
            title,
            artifact(config, f"maps/{filename}.html"),
            cmap,
            norm,
            unit,
            config["maps"]["tiles"],
        )
        manifest["maps"].append(
            {
                "file": path.name,
                "title": title,
                "unit": unit,
                "range": [norm.vmin, norm.vmax],
                "sha256": sha256(path),
            }
        )

    def draw_population(ax):
        style_axis(ax, boundary, extent)
        # Robust display cap only: raw numerical values remain in the artifacts.
        finite = population.projected_change_pct.dropna()
        cap = max(float(np.quantile(np.abs(finite), 0.98)), 1)
        cmap, norm = color_scale(finite, divergent=True, limit=cap)
        population.plot(
            column="projected_change_pct",
            ax=ax,
            cmap=cmap,
            norm=norm,
            linewidth=0,
            zorder=1,
            missing_kwds={"color": "#d1d5db"},
        )
        bar = ax.figure.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            fraction=0.042,
            pad=0.03,
            extend="both",
        )
        bar.set_label("Projected change (%); display capped at abs. 98th percentile", fontsize=7)
        bar.ax.tick_params(labelsize=8)
        ax.set_title(
            "Population scenario | 2010 to 2020\n2017 projection; NOT observed population change",
            loc="left",
            fontsize=10,
            fontweight="bold",
        )

    def draw_railway(ax):
        style_axis(ax, boundary, extent)
        railway.plot(ax=ax, color="#5b6285", linewidth=0.65, zorder=2)
        stations.plot(ax=ax, color="#ca5a30", markersize=4, zorder=3)
        ax.set_title(
            f"Railway context | {config['datasets']['railway']['year']}\nStation sections merged by name and proximity",
            loc="left",
            fontsize=10,
            fontweight="bold",
        )

    for filename, renderer in [
        ("population_change_map", draw_population),
        ("railway_map", draw_railway),
    ]:
        fig, ax = plt.subplots(figsize=(8, 9))
        renderer(ax)
        fig.text(
            0.05, 0.025, ATTRIBUTION + "\nTime periods differ from the land-price maps.", fontsize=7
        )
        fig.tight_layout(rect=[0, 0.055, 1, 1])
        path = artifact(config, f"figures/{filename}.png")
        fig.savefig(path, dpi=config["maps"]["dpi"])
        plt.close(fig)
        manifest["maps"].append({"file": path.name, "sha256": sha256(path)})
    fig, axes = plt.subplots(2, 2, figsize=(13, 14))
    for ax, spec in [(axes[0, 0], specifications[0]), (axes[1, 1], specifications[2])]:
        _, points, column, title, unit, scale = spec
        cmap, norm = color_scale(points[column], divergent=True, limit=scale)
        draw_points(ax, points, column, title, boundary, extent, cmap, norm, unit, 6)
    draw_population(axes[0, 1])
    draw_railway(axes[1, 0])
    fig.suptitle(
        "Osaka | Spatial comparison with different reference periods",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.018,
        ATTRIBUTION
        + "\nVisual correspondence does not establish causation. All panels use the same extent and projected CRS.",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(artifact(config, "figures/vlm_input_overview.png"), dpi=config["maps"]["dpi"])
    plt.close(fig)
    write_json(artifact(config, "maps/map_manifest.json"), manifest)
    return artifact(config, "figures/vlm_input_overview.png")
