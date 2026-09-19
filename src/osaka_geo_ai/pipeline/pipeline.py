"""Small stage orchestrator; domain work lives in the imported stage modules."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from osaka_geo_ai.config import artifact, directory
from osaka_geo_ai.io import sha256, write_json

LOG = logging.getLogger(__name__)
STAGES = {
    "download": ("data.downloader", "download"),
    "prepare": ("data.preprocessing", "prepare"),
    "basic_maps": ("gis.visualization", "build_basic_maps"),
    "features": ("gis.features", "build_features"),
    "train": ("models.train", "train"),
    "predict": ("models.predict", "predict"),
    "maps": ("gis.visualization", "build_maps"),
    "vlm": ("vision.vlm", "analyze_maps"),
    "report": ("llm.reporter", "generate_report"),
}
PHASE_STAGES = {
    1: ["download", "prepare", "basic_maps"],
    2: ["download", "prepare", "features", "train", "predict", "maps"],
    3: ["download", "prepare", "features", "train", "predict", "maps", "vlm"],
    4: ["download", "prepare", "features", "train", "predict", "maps", "vlm", "report"],
}


def run_stage(stage, config, **kwargs):
    module, function = STAGES[stage]
    return getattr(importlib.import_module("osaka_geo_ai." + module), function)(config, **kwargs)


def run_pipeline(config, *, phase=4, offline=False, refresh=False):
    root = Path(config["_root"])
    metadata = {
        "schema_version": "1.0",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "config": config,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "source_sha256": {
            str(p.relative_to(root)): sha256(p) for p in sorted((root / "src").rglob("*.py"))
        },
        "stages": [],
    }
    path = artifact(config, "run_metadata.json")
    write_json(path, metadata)
    try:
        for stage in PHASE_STAGES[phase]:
            LOG.info("Stage: %s", stage)
            start = time.monotonic()
            record = {"name": stage, "status": "running"}
            metadata["stages"].append(record)
            write_json(path, metadata)
            result = run_stage(
                stage,
                config,
                **({"offline": offline, "refresh": refresh} if stage == "download" else {}),
            )
            record.update(
                status="completed",
                elapsed_seconds=round(time.monotonic() - start, 3),
                artifact=str(result.relative_to(root)),
                sha256=sha256(result),
            )
            write_json(path, metadata)
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        if metadata["stages"]:
            metadata["stages"][-1]["status"] = "failed"
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest = directory(config, "raw") / "manifest.json"
        if manifest.exists():
            metadata["download_manifest_sha256"] = sha256(manifest)
        write_json(path, metadata)
    LOG.info("Pipeline complete: %s", artifact(config, "reports/report.md"))
    return path
