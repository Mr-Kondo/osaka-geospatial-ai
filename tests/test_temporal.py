import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from osaka_geo_ai.config import validate_config
from osaka_geo_ai.gis.features import link_annual_prices
from osaka_geo_ai.models.evaluate import regression_metrics, temporal_fold
from osaka_geo_ai.models.train import make_models


def panel():
    return gpd.GeoDataFrame(
        {
            "row_id": ["2021:a", "2022:b", "2023:b", "2024:b"],
            "year": [2021, 2022, 2023, 2024],
            "site_id": ["a", "b", "b", "b"],
            "previous_site_id": ["none", "a", "b", "b"],
            "selection": [4, 2, 1, 1],
            "use_code": ["000"] * 4,
            "land_price": [100.0, 110.0, 121.0, 150.0],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0), Point(200, 0)],
        crs="EPSG:6674",
    )


def test_renumbering_lags_and_moved_site_exclusion():
    f, counts = link_annual_prices(panel(), 50)
    assert f.loc[0, "next_year_land_price_change_rate"] == pytest.approx(0.1)
    assert f.loc[2, "previous_land_price_change"] == pytest.approx(0.1)
    assert np.isnan(f.loc[2, "next_year_land_price_change_rate"])
    assert counts[-1]["excluded_movement_or_use"] == 1


def test_changing_future_price_cannot_change_earlier_predictors():
    original, _ = link_annual_prices(panel(), 50)
    changed = panel()
    changed.loc[2, "land_price"] = 500
    updated, _ = link_annual_prices(changed, 50)
    cols = [
        "land_price",
        "previous_land_price",
        "land_price_change_rate",
        "previous_land_price_change",
    ]
    pd.testing.assert_frame_equal(original.loc[:1, cols], updated.loc[:1, cols])
    assert (
        original.loc[1, "next_year_land_price_change_rate"]
        != updated.loc[1, "next_year_land_price_change_rate"]
    )


def test_gaps_do_not_become_next_year_labels():
    f, _ = link_annual_prices(panel().iloc[[0, 2]], 50)
    assert f.next_year_land_price_change_rate.isna().all()


def test_temporal_fold_and_feature_leak_guard(config):
    frame = pd.DataFrame(
        {
            "year": [2021, 2022, 2023, 2024],
            "target_year": [2022, 2023, 2024, 2025],
            "target": [0.1, 0.2, 0.3, 0.4],
        }
    )
    train, test = temporal_fold(frame, 2024, "target", 2021)
    assert train.target_year.max() <= test.year.min()
    config["model"]["numeric_features"].append("next_year_land_price")
    with pytest.raises(ValueError, match="Future"):
        validate_config(config)


@pytest.mark.parametrize("name", ["dummy", "ridge", "random_forest", "lightgbm"])
def test_model_input_output_shape_and_train_only_imputation(config, name):
    spec = config["model"]
    spec["random_forest"]["n_estimators"] = 3
    spec["lightgbm"]["n_estimators"] = 3
    rng = np.random.default_rng(42)
    x = pd.DataFrame(
        rng.normal(size=(50, len(spec["numeric_features"]))), columns=spec["numeric_features"]
    )
    for col in spec["categorical_features"]:
        x[col] = "a"
    x.loc[0, spec["numeric_features"][0]] = np.nan
    model = make_models(config)[name]
    model.fit(x.iloc[:40], rng.normal(size=40))
    future = x.iloc[40:].copy()
    future[spec["categorical_features"][0]] = "unseen"
    future[spec["numeric_features"][0]] = 1000000
    before = (
        model.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["impute"]
        .statistics_.copy()
    )
    result = model.predict(future)
    assert result.shape == (10,) and np.isfinite(result).all()
    np.testing.assert_equal(
        before,
        model.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["impute"]
        .statistics_,
    )


def test_metrics_units_and_direction():
    result = regression_metrics([0.05, -0.02, 0.0], [0.03, -0.01, 0.0])
    assert result["mae"] == pytest.approx(0.01)
    assert result["direction_accuracy"] == 1
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1])
