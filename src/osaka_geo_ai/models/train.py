"""Compare four tabular models; choose on validation only and then freeze for test."""

from __future__ import annotations

import logging

import joblib
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from osaka_geo_ai.config import artifact, directory
from osaka_geo_ai.data.loader import read_geo
from osaka_geo_ai.data.validation import require_columns
from osaka_geo_ai.io import sha256, write_json
from osaka_geo_ai.models.evaluate import regression_metrics, temporal_fold

LOG = logging.getLogger(__name__)


def make_models(config):
    spec, seed = config["model"], config["project"]["seed"]
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="median", add_indicator=True, keep_empty_features=True
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                spec["numeric_features"],
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                spec["categorical_features"],
            ),
        ],
        remainder="drop",
    ).set_output(transform="pandas")
    estimators = {
        "dummy": DummyRegressor(strategy="mean"),
        "ridge": Ridge(**spec["ridge"]),
        "random_forest": RandomForestRegressor(random_state=seed, **spec["random_forest"]),
        "lightgbm": LGBMRegressor(random_state=seed, **spec["lightgbm"]),
    }
    return {
        name: Pipeline([("preprocess", clone(preprocess)), ("regressor", estimator)])
        for name, estimator in estimators.items()
    }


def train(config):
    feature_path = directory(config, "processed") / "features.parquet"
    frame = read_geo(feature_path)
    spec = config["model"]
    target = spec["target"]
    columns = spec["numeric_features"] + spec["categorical_features"]
    require_columns(frame, columns + [target, "target_year"])
    first = config["analysis"]["first_origin_year"]
    training, validation = temporal_fold(frame, spec["validation_target_year"], target, first)
    candidates, scores = make_models(config), {}
    for name, estimator in candidates.items():
        LOG.info("Training %s; validation target %s", name, spec["validation_target_year"])
        estimator.fit(training[columns], training[target])
        scores[name] = regression_metrics(
            validation[target], estimator.predict(validation[columns])
        )
        LOG.info("%s validation MAE=%.6f", name, scores[name]["mae"])
    chosen = min(scores, key=lambda name: (scores[name][spec["selection_metric"]], name))
    refit, test = temporal_fold(frame, spec["test_target_year"], target, first)
    if refit.target_year.max() > spec["validation_target_year"]:
        raise ValueError("Test refit includes labels later than the configured validation cutoff")
    tests = {}
    model_dir = artifact(config, "models")
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, estimator in make_models(config).items():
        estimator.fit(refit[columns], refit[target])
        tests[name] = regression_metrics(test[target], estimator.predict(test[columns]))
        if name == chosen:
            joblib.dump(
                {
                    "estimator": estimator,
                    "features": columns,
                    "target": target,
                    "name": chosen,
                    "fit_max_target_year": int(refit.target_year.max()),
                    "features_sha256": sha256(feature_path),
                },
                model_dir / "model.joblib",
            )
    rolling = []
    for year in spec.get("rolling_target_years", []):
        past, held_out = temporal_fold(frame, year, target, first)
        for name, estimator in make_models(config).items():
            estimator.fit(past[columns], past[target])
            rolling.append(
                {
                    "model": name,
                    "target_year": year,
                    "fit_max_target_year": int(past.target_year.max()),
                    **regression_metrics(held_out[target], estimator.predict(held_out[columns])),
                }
            )
    # Separate deployment fit: it cannot contaminate the saved held-out test model.
    latest = int(frame.year.max())
    known = frame[frame[target].notna() & frame.year.ge(first) & frame.target_year.le(latest)]
    forecast_model = make_models(config)[chosen]
    forecast_model.fit(known[columns], known[target])
    joblib.dump(
        {
            "estimator": forecast_model,
            "features": columns,
            "target": target,
            "name": chosen,
            "fit_max_target_year": int(known.target_year.max()),
            "features_sha256": sha256(feature_path),
        },
        model_dir / "forecast_model.joblib",
    )
    result = {
        "schema_version": "1.0",
        "target": target,
        "target_unit": "fraction; multiply MAE/RMSE by 100 for percentage points",
        "selected_model": chosen,
        "selection": "lowest validation MAE only; test is not used for selection",
        "validation_target_year": spec["validation_target_year"],
        "test_target_year": spec["test_target_year"],
        "train_target_years": sorted(int(x) for x in training.target_year.unique()),
        "refit_target_years": sorted(int(x) for x in refit.target_year.unique()),
        "validation": scores,
        "test": tests,
        "selected_test_metrics": tests[chosen],
        "rolling_backtest": rolling,
        "features": columns,
        "features_sha256": sha256(feature_path),
    }
    write_json(artifact(config, "metrics/metrics.json"), result)
    LOG.info("Selected %s; test MAE=%.6f", chosen, tests[chosen]["mae"])
    return artifact(config, "metrics/metrics.json")
