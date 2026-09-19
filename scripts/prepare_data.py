#!/usr/bin/env python3
"""CLI entry point; execution logic is in osaka_geo_ai."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from osaka_geo_ai.cli import main

if __name__ == "__main__":
    main("prepare")
