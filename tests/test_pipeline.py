from pathlib import Path

import pytest

from osaka_geo_ai.io import read_json
from osaka_geo_ai.pipeline import pipeline


def write_stage_artifact(stage, config, **_kwargs):
    artifact_path = Path(config["_root"]) / f"{stage}.txt"
    artifact_path.write_text(stage)
    return artifact_path


def test_pipeline_progress_reports_completed_stages(isolated_config, monkeypatch, capsys):
    monkeypatch.setitem(pipeline.PHASE_STAGES, 1, ["download", "prepare"])
    monkeypatch.setattr(pipeline, "run_stage", write_stage_artifact)

    metadata_path = pipeline.run_pipeline(isolated_config, phase=1)

    progress_output = capsys.readouterr().err
    metadata = read_json(metadata_path)
    assert "Osaka pipeline" in progress_output and "100%" in progress_output
    assert [stage["status"] for stage in metadata["stages"]] == ["completed", "completed"]


def test_pipeline_progress_records_failed_stage(isolated_config, monkeypatch, capsys):
    def fail_stage(stage, config, **kwargs):
        if stage == "prepare":
            raise RuntimeError("stage failed")
        return write_stage_artifact(stage, config, **kwargs)

    monkeypatch.setitem(pipeline.PHASE_STAGES, 1, ["download", "prepare"])
    monkeypatch.setattr(pipeline, "run_stage", fail_stage)

    with pytest.raises(RuntimeError, match="stage failed"):
        pipeline.run_pipeline(isolated_config, phase=1)

    progress_output = capsys.readouterr().err
    metadata = read_json(Path(isolated_config["_root"]) / "artifacts/run_metadata.json")
    assert "Osaka pipeline failed: prepare" in progress_output
    assert metadata["status"] == "failed"
    assert metadata["stages"][-1]["status"] == "failed"
    assert metadata["stages"][-1]["elapsed_seconds"] >= 0
