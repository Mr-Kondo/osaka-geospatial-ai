"""Download only configured public URLs; cache and verify archives without scraping HTML."""

from __future__ import annotations

import logging
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from osaka_geo_ai.config import directory
from osaka_geo_ai.io import read_json, sha256, write_json

LOG = logging.getLogger(__name__)


def validate_archive(path: Path, max_expanded_mb: int):
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Expected ZIP, received another content type: {path}")
    with zipfile.ZipFile(path) as archive:
        if sum(x.file_size for x in archive.infolist()) > max_expanded_mb * 1024**2:
            raise ValueError(f"Archive expands beyond configured limit: {path}")
        for info in archive.infolist():
            name = Path(info.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Corrupt ZIP member: {bad}")


def download(config: dict, *, offline=False, refresh=False):
    settings = config["download"]
    session = requests.Session()
    session.headers["User-Agent"] = "osaka-geospatial-ai/0.1 (research PoC)"
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=settings["retries"],
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
        ),
    )
    root = directory(config, "raw")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    previous = read_json(manifest_path) if manifest_path.exists() else {"files": []}
    previous = {v["path"]: v for v in previous["files"]}
    entries = []
    for name, dataset in config["datasets"].items():
        if not dataset.get("enabled", True):
            continue
        for item in dataset.get("files", []):
            destination = root / name / item["filename"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            key = str(destination.relative_to(root))
            old = previous.get(key)
            if offline and not destination.exists():
                raise FileNotFoundError(
                    f"Offline: missing {destination}. Download from {dataset['source']} and place it here."
                )
            if destination.exists() and not refresh:
                LOG.info("Using cached %s", key)
                if old and old["sha256"] != sha256(destination):
                    raise ValueError(
                        f"Cached checksum changed: {destination}; use --refresh to redownload"
                    )
            elif not offline:
                LOG.info("Downloading %s", key)
                partial = destination.with_suffix(".part")
                try:
                    with session.get(
                        item["url"], stream=True, timeout=settings["timeout_seconds"]
                    ) as response:
                        response.raise_for_status()
                        size = 0
                        with partial.open("wb") as stream:
                            for chunk in response.iter_content(1024 * 256):
                                size += len(chunk)
                                if size > settings["max_archive_mb"] * 1024**2:
                                    raise ValueError("Download exceeds max_archive_mb")
                                stream.write(chunk)
                    validate_archive(partial, settings["max_expanded_mb"])
                    partial.replace(destination)
                    old = None
                except (requests.RequestException, ValueError) as exc:
                    raise RuntimeError(
                        f"Download failed: {item['url']}. Manual source: {dataset['source']}; place at {destination}. {exc}"
                    ) from exc
                finally:
                    partial.unlink(missing_ok=True)
            validate_archive(destination, settings["max_expanded_mb"])
            digest = sha256(destination)
            if item.get("sha256") and digest != item["sha256"]:
                raise ValueError(f"Pinned checksum mismatch: {destination}")
            entries.append(
                {
                    "dataset": name,
                    "path": key,
                    "url": item["url"],
                    "sha256": digest,
                    "bytes": destination.stat().st_size,
                    "retrieved_at": old["retrieved_at"]
                    if old
                    else datetime.now(timezone.utc).isoformat(),
                    "source": dataset["source"],
                    "license": dataset["license"],
                    "year": item.get("year", dataset.get("year")),
                }
            )
    write_json(manifest_path, {"schema_version": "1.0", "files": entries})
    session.close()
    return manifest_path
