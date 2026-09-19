"""Read-only artifact display adapters for the presentation-only Colab notebook."""

from pathlib import Path

import pandas as pd

from osaka_geo_ai.io import read_json


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
