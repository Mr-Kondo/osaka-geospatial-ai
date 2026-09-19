"""Explicit archive members and column mappings, not format guesses."""

from __future__ import annotations

import fnmatch
import zipfile
from pathlib import Path

import geopandas as gpd

from osaka_geo_ai.data.downloader import validate_archive
from osaka_geo_ai.data.validation import validate_geo
from osaka_geo_ai.io import sha256


def read_archive(
    path: Path, pattern: str, *, columns=None, bbox=None, allow_empty=False
) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/download_data.py first")
    validate_archive(path, 2048)
    with zipfile.ZipFile(path) as archive:
        matches = sorted(
            n
            for n in archive.namelist()
            if fnmatch.fnmatch(n, pattern) and not n.startswith("__MACOSX/")
        )
        if len(matches) != 1:
            raise ValueError(f"Expected one member matching {pattern} in {path}, found {matches}")
        # Some official archives use backslashes. Normalize only the selected shape's siblings.
        member = matches[0]
        target_dir = path.parent / "_extracted" / path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        digest = sha256(path)
        stamp = target_dir / ".sha256"
        target = target_dir / Path(member.replace("\\", "/")).name
        if not target.exists() or not stamp.exists() or stamp.read_text() != digest:
            stem = member.rsplit(".", 1)[0]
            for name in archive.namelist():
                if name.rsplit(".", 1)[0] == stem and name.rsplit(".", 1)[-1].lower() in {
                    "shp",
                    "shx",
                    "dbf",
                    "prj",
                    "cpg",
                }:
                    (target_dir / Path(name.replace("\\", "/")).name).write_bytes(
                        archive.read(name)
                    )
            stamp.write_text(digest)
    encoding = (
        target.with_suffix(".cpg").read_text().strip()
        if target.with_suffix(".cpg").exists()
        else "CP932"
    )
    frame = gpd.read_file(target, columns=columns, bbox=bbox, engine="pyogrio", encoding=encoding)
    if frame.empty and allow_empty and frame.crs is not None:
        return frame
    validate_geo(frame, path.name)
    return frame


def read_geo(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing stage artifact {path}; run its preceding stage")
    frame = gpd.read_parquet(path)
    validate_geo(frame, str(path))
    return frame
