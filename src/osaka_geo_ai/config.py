"""Configuration loading; all paths are relative to the repository, not the shell cwd."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml
from pyproj import CRS

ROOT = Path(__file__).resolve().parents[2]


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path = ROOT / "configs/default.yaml", _seen=None) -> dict:
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Cyclic config include: {path}")
    seen.add(path)
    with path.open(encoding="utf-8") as stream:
        own = yaml.safe_load(stream) or {}
    config = {}
    for include in own.pop("includes", []):
        config = merge(config, load_config(path.parent / include, seen.copy()))
    config = merge(config, own)
    if _seen is None:
        config["_root"] = str(ROOT)
        config["_config_path"] = str(path)
        validate_config(config)
    return config


def validate_config(config: dict) -> None:
    crs = CRS(config["analysis"]["analysis_crs"])
    if not crs.is_projected or any(a.unit_name != "metre" for a in crs.axis_info):
        raise ValueError("analysis_crs must be a projected CRS in metres")
    if config["analysis"]["station_radius_m"] != 1000:
        raise ValueError("station_radius_m must be 1000 to preserve the station_count_1km contract")
    model = config["model"]
    if model["target"] != "next_year_land_price_change_rate":
        raise ValueError("This version implements next_year_land_price_change_rate only")
    if (
        not config["analysis"]["first_origin_year"]
        < model["validation_target_year"]
        < model["test_target_year"]
    ):
        raise ValueError("Need train origins < validation target year < test target year")
    if config["analysis"]["target_year"] != model["test_target_year"]:
        raise ValueError("analysis.target_year must equal model.test_target_year")
    known = set(config["model"]["numeric_features"] + config["model"]["categorical_features"])
    forbidden = {model["target"], "target_year", "next_year_land_price", "actual", "residual"}
    if known & forbidden or any(x.startswith("next_") for x in known):
        raise ValueError("Future/target columns cannot be model inputs")


def directory(config: dict, key: str) -> Path:
    return Path(config["_root"]) / config["paths"][key]


def artifact(config: dict, relative: str) -> Path:
    return directory(config, "artifacts") / relative
