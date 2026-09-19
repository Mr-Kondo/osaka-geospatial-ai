"""Human-facing interactive maps, kept separate from static VLM inputs."""

import html

import folium
import numpy as np
from branca.colormap import LinearColormap
from matplotlib.colors import to_hex


def interactive_map(points, boundary, column, title, path, cmap, norm, unit, tiles=None):
    geographic = points.to_crs("EPSG:4326")
    center = [float(geographic.geometry.y.mean()), float(geographic.geometry.x.mean())]
    result = folium.Map(location=center, zoom_start=10, tiles=tiles, prefer_canvas=True)
    folium.GeoJson(
        boundary.to_crs("EPSG:4326").to_json(),
        style_function=lambda _: {"color": "#667085", "weight": 0.7, "fillOpacity": 0.025},
    ).add_to(result)
    for _, row in geographic.iterrows():
        value = row[column]
        if not np.isfinite(value):
            continue
        color = to_hex(cmap(norm(value)))
        text = f"{html.escape(str(row.get('site_id', '')))} | {html.escape(str(row.get('municipality_name', '')))}<br>{html.escape(title)}: {value:.3f} {unit}"
        folium.CircleMarker(
            [row.geometry.y, row.geometry.x],
            radius=3.5,
            weight=0.4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=folium.Tooltip(text),
        ).add_to(result)
    ticks = np.linspace(norm.vmin, norm.vmax, 9)
    LinearColormap(
        [to_hex(cmap(norm(v))) for v in ticks],
        index=ticks,
        vmin=norm.vmin,
        vmax=norm.vmax,
        caption=f"{title} ({unit})",
    ).add_to(result)
    result.get_root().html.add_child(
        folium.Element(
            f'<div style="position:fixed;bottom:14px;left:14px;z-index:9999;background:white;padding:10px;font:12px sans-serif">{html.escape(title)}<br>Source: MLIT National Land Numerical Information; processed by this PoC</div>'
        )
    )
    bounds = geographic.total_bounds
    result.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(path))
