"""Read-only artifact display adapters for the presentation-only Colab notebook."""

from pathlib import Path

import pandas as pd

from osaka_geo_ai.config import load_config
from osaka_geo_ai.io import read_json


def configuration_summary(root=Path("."), config_path="configs/default.yaml"):
    root = Path(root)
    config = load_config(root / config_path)
    try:
        import torch

        if torch.cuda.is_available():
            runtime = f"CUDA: {torch.cuda.get_device_name(0)}"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            runtime = "Apple MPS"
        else:
            runtime = "CPU; AI stages will report unavailable unless CPU is explicitly enabled"
    except ImportError:
        runtime = "PyTorch unavailable; install the ai extra"
    return pd.DataFrame(
        [
            {"item": "config", "value": str(Path(config_path))},
            {"item": "runtime accelerator", "value": runtime},
            {
                "item": "VLM",
                "value": f"{config['vlm']['model']} ({config['vlm']['device']})",
            },
            {
                "item": "LLM",
                "value": f"{config['llm']['model']} ({config['llm']['device']})",
            },
            {
                "item": "GIS data",
                "value": "official URLs; automatic download with SHA-256 verification",
            },
        ]
    )


def dataset_summary(root=Path(".")):
    summary = read_json(Path(root) / "artifacts/dataset_summary.json")
    return pd.DataFrame(
        [
            {"year": int(k), "land_price_sites": v}
            for k, v in summary["land_price_rows_by_year"].items()
        ]
    )


def metrics_table(root=Path(".")):
    metrics = read_json(Path(root) / "artifacts/metrics/metrics.json")
    return pd.DataFrame(
        [
            {
                "split": split,
                "model": name,
                **values,
                "mae_pp": values["mae"] * 100,
                "rmse_pp": values["rmse"] * 100,
            }
            for split in ("validation", "test")
            for name, values in metrics[split].items()
        ]
    )


def prediction_preview(root=Path("."), count=15):
    frame = pd.read_parquet(Path(root) / "artifacts/predictions/predictions.parquet")
    return frame.drop(columns="geometry").head(count)
