import ast
import json
import zipfile
from pathlib import Path

import nbformat
import pytest
from pydantic import ValidationError

from osaka_geo_ai.config import artifact, load_config
from osaka_geo_ai.data.downloader import validate_archive
from osaka_geo_ai.io import read_json, write_json
from osaka_geo_ai.llm.schemas import VisualPattern, VLMAnalysis
from osaka_geo_ai.models.predict import validate_prediction_contract
from osaka_geo_ai.presentation import configuration_summary
from osaka_geo_ai.vision.vlm import ProviderUnavailable, analyze_maps, parse_findings


def test_structured_output_rejects_confidence_outside_range_and_nan():
    for value in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValidationError):
            VisualPattern(region="north", observation="cluster", confidence=value)
    with pytest.raises(ValidationError):
        VisualPattern(region="north", observation="cluster", confidence=0.5, unknown=True)


def test_disabled_vlm_cannot_claim_findings():
    with pytest.raises(ValidationError, match="no visual findings"):
        VLMAnalysis(
            status="disabled",
            provider="hf",
            model="test",
            input_images=[],
            visual_patterns=[{"region": "north", "observation": "cluster", "confidence": 0.5}],
            anomalies=[],
            cross_map_relationships=[],
            limitations=[],
        )


def test_json_parser_does_not_accept_free_prose():
    with pytest.raises(json.JSONDecodeError):
        parse_findings("I think there is a cluster")
    assert (
        parse_findings(
            '```json\n{"visual_patterns":[],"anomalies":[],"cross_map_relationships":[],"limitations":[]}\n```'
        ).visual_patterns
        == []
    )


@pytest.mark.parametrize(
    "failure",
    [ProviderUnavailable("no GPU"), RuntimeError("out of memory"), ValueError("invalid output")],
)
def test_provider_failure_keeps_explicit_empty_artifact(isolated_config, failure):
    c = isolated_config
    c["vlm"]["enabled"] = True
    for name in c["vlm"]["images"]:
        path = artifact(c, "figures/" + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    write_json(artifact(c, "maps/map_manifest.json"), {})

    class FailedProvider:
        def analyze(self, images, prompt):
            raise failure

    result = VLMAnalysis.model_validate(read_json(analyze_maps(c, FailedProvider())))
    assert result.status in {"unavailable", "failed"} and not result.visual_patterns


def test_provider_result_validated_and_images_hashed(isolated_config):
    c = isolated_config
    c["vlm"]["enabled"] = True
    for name in c["vlm"]["images"]:
        path = artifact(c, "figures/" + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    write_json(artifact(c, "maps/map_manifest.json"), {})

    class Provider:
        def analyze(self, images, prompt):
            return json.dumps(
                {
                    "visual_patterns": [
                        {"region": "north", "observation": "red concentration", "confidence": 0.4}
                    ],
                    "anomalies": [],
                    "cross_map_relationships": [],
                    "limitations": ["test double"],
                }
            )

    result = read_json(analyze_maps(c, Provider()))
    assert result["status"] == "completed" and len(result["input_images"][0]["sha256"]) == 64


def test_zip_slip_rejected(tmp_path):
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        validate_archive(path, 10)


def test_config_cycle_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("includes: [b.yaml]")
    (tmp_path / "b.yaml").write_text("includes: [a.yaml]")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(tmp_path / "a.yaml")


def test_stale_prediction_contract_rejected(isolated_config):
    c = isolated_config
    write_json(
        artifact(c, "predictions/prediction_metadata.json"),
        {"inputs": [{"path": "missing.parquet", "sha256": "bad"}], "outputs": []},
    )
    with pytest.raises(ValueError, match="contract changed"):
        validate_prediction_contract(c)


def test_notebook_contains_presentation_only():
    notebook = nbformat.read(
        Path(__file__).resolve().parents[1] / "notebooks/colab_demo.ipynb", as_version=4
    )
    nbformat.validate(notebook)
    forbidden = {"fit", "sjoin", "buffer", "read_file", "to_crs", "predict", "train_test_split"}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        assert not any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for n in ast.walk(tree)
        )
        assert not any(isinstance(n, ast.Attribute) and n.attr in forbidden for n in ast.walk(tree))
        domain_imports = [
            n.module
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
            and n.module is not None
            and n.module.startswith("osaka_geo_ai")
        ]
        assert all(module == "osaka_geo_ai.presentation" for module in domain_imports)


def test_colab_notebook_clones_public_repository_and_runs_pipeline():
    notebook = nbformat.read(
        Path(__file__).resolve().parents[1] / "notebooks/colab_demo.ipynb", as_version=4
    )
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "https://github.com/Mr-Kondo/osaka-geospatial-ai.git" in source
    assert 'find_spec("google") is not None and find_spec("google.colab") is not None' in source
    assert 'sys.path.insert(0, str(ROOT / "src"))' in source
    assert '"git", "clone"' in source
    assert '"checkout", "--detach", "FETCH_HEAD"' in source
    assert '"git", "pull"' not in source
    assert '"scripts/run_pipeline.py"' in source
    assert '"configs/colab.yaml"' in source
    assert "osaka-geospatial-ai.zip" not in source
    assert "ENABLE_AI" not in source


def test_colab_config_enables_huggingface_models_and_online_downloads():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/colab.yaml")
    assert config["vlm"]["enabled"] and config["vlm"]["provider"] == "huggingface"
    assert config["llm"]["enabled"] and config["llm"]["provider"] == "huggingface"
    assert not config["vlm"]["local_files_only"]
    assert not config["llm"]["local_files_only"]
    assert config["datasets"]["land_price"]["enabled"]
    assert config["datasets"]["population"]["enabled"]
    assert config["datasets"]["railway"]["enabled"]
    assert config["vlm"]["device"] == "cuda"
    assert config["llm"]["device"] == "cuda"


def test_local_ai_config_auto_selects_cuda_or_apple_mps():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/ai.yaml")
    assert config["vlm"]["device"] == "auto"
    assert config["llm"]["device"] == "auto"


def test_configuration_summary_is_read_only_presentation_data():
    root = Path(__file__).resolve().parents[1]
    summary = configuration_summary(root, "configs/colab.yaml")
    assert list(summary.columns) == ["item", "value"]
    assert set(summary["item"]) == {"config", "runtime accelerator", "VLM", "LLM", "GIS data"}
    assert "Qwen/Qwen2-VL-2B-Instruct" in summary.loc[summary["item"] == "VLM", "value"].item()
