import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from osaka_geo_ai.data.validation import require_columns, validate_geo
from osaka_geo_ai.gis.spatial import (
    flood_features,
    landuse_features,
    polygon_features,
    station_features,
)


def points(coords, crs="EPSG:6674"):
    return gpd.GeoDataFrame(geometry=[Point(*p) for p in coords], crs=crs)


def test_missing_and_geographic_crs_rejected():
    with pytest.raises(ValueError, match="CRS is missing"):
        validate_geo(gpd.GeoDataFrame(geometry=[Point(0, 0)]))
    with pytest.raises(ValueError, match="metres"):
        station_features(points([(135.5, 34.7)], "EPSG:4326"), points([(135.6, 34.7)], "EPSG:4326"))
    with pytest.raises(ValueError, match="required columns"):
        require_columns(points([(0, 0)]), ["population"])


def test_exact_distance_and_inclusive_radius():
    result = station_features(
        points([(0, 0), (0, 2000)]), points([(300, 400), (1000, 0), (1001, 0)])
    )
    assert result.iloc[0].nearest_station_distance_m == 500
    assert result.iloc[0].station_count_1km == 2
    assert result.iloc[1].station_count_1km == 0


def test_osaka_projection_has_metre_scale():
    a = points([(135.5, 34.7)], "EPSG:4326").to_crs("EPSG:6674")
    b = points([(135.51, 34.7)], "EPSG:4326").to_crs("EPSG:6674")
    assert 910 < station_features(a, b).nearest_station_distance_m.iloc[0] < 920


def test_mesh_edge_is_deterministic_and_missing_remains_missing():
    meshes = gpd.GeoDataFrame(
        {"mesh_id": ["2", "1"], "population": [20, 10]},
        geometry=[box(0, -2, 2, 2), box(-2, -2, 0, 2)],
        crs="EPSG:6674",
    )
    result = polygon_features(points([(0, 0), (10, 10)]), meshes, ["population"], "mesh_id")
    assert result.population.iloc[0] == 10
    assert np.isnan(result.population.iloc[1])


def test_area_weighted_landuse_and_coverage():
    mesh = gpd.GeoDataFrame(
        {"landuse_code": ["0700", "0500"]},
        geometry=[box(-200, -200, 0, 200), box(0, -200, 200, 200)],
        crs="EPSG:6674",
    )
    result = landuse_features(
        points([(0, 0), (500, 0)]), mesh, {"built_up": ["0700"], "forest": ["0500"]}, 100
    )
    assert result.built_up_ratio.iloc[0] == pytest.approx(0.5)
    assert result.forest_ratio.iloc[0] == pytest.approx(0.5)
    assert np.isnan(result.built_up_ratio.iloc[1])
    assert result.landuse_coverage_ratio.iloc[1] == 0


def test_unknown_flood_coverage_is_not_zero_risk():
    flood = gpd.GeoDataFrame(
        {"flood_depth_m": [2.0, 3.0]}, geometry=[box(0, 0, 2, 2), box(1, 1, 3, 3)], crs="EPSG:6674"
    )
    coverage = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:6674")
    result = flood_features(points([(1.5, 1.5), (5, 5), (20, 20)]), flood, coverage)
    assert result.flood_depth_m.tolist()[:2] == [3.0, 0.0]
    assert result.flood_risk.tolist()[:2] == [1.0, 0.0]
    assert np.isnan(result.flood_risk.iloc[2])
