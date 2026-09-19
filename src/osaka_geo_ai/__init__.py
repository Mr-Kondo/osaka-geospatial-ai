"""Osaka Geospatial AI: explicit artifacts and evidence boundaries."""

import os
from pathlib import Path

# Matplotlib can also be imported by model libraries before the visualization stage.
_cache = Path(__file__).resolve().parents[2] / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache))

__version__ = "0.1.0"
