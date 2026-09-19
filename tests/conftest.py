import copy
from pathlib import Path

import pytest

from osaka_geo_ai.config import load_config


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")


@pytest.fixture
def isolated_config(config, tmp_path):
    result = copy.deepcopy(config)
    result["_root"] = str(tmp_path)
    return result
