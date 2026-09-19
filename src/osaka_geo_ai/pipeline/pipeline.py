"""Small stage orchestrator; domain work lives in the imported stage modules."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

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
    metadata_path = artifact(config, "run_metadata.json")
    write_json(metadata_path, metadata)
    progress = tqdm(
        PHASE_STAGES[phase], desc="Osaka pipeline", unit="stage", dynamic_ncols=True
    )
    try:
        with logging_redirect_tqdm():
            for stage in progress:
                progress.set_postfix_str(stage)
                LOG.info("Stage: %s", stage)
                stage_started_at = time.monotonic()
                stage_record = {"name": stage, "status": "running"}
                metadata["stages"].append(stage_record)
                write_json(metadata_path, metadata)
                stage_artifact = run_stage(
                    stage,
                    config,
                    **({"offline": offline, "refresh": refresh} if stage == "download" else {}),
                )
                stage_record.update(
                    status="completed",
                    elapsed_seconds=round(time.monotonic() - stage_started_at, 3),
                    artifact=str(stage_artifact.relative_to(root)),
                    sha256=sha256(stage_artifact),
                )
                progress.set_postfix_str(f"{stage}: {stage_record['elapsed_seconds']:.1f}s")
                write_json(metadata_path, metadata)
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        if metadata["stages"]:
            metadata["stages"][-1].update(
                status="failed", elapsed_seconds=round(time.monotonic() - stage_started_at, 3)
            )
            progress.write(f"Osaka pipeline failed: {stage}", file=progress.fp)
        raise
    finally:
        progress.close()
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest = directory(config, "raw") / "manifest.json"
        if manifest.exists():
            metadata["download_manifest_sha256"] = sha256(manifest)
        write_json(metadata_path, metadata)
    LOG.info("Pipeline complete: %s", artifact(config, "reports/report.md"))
    return metadata_path
