# Verification Record

- Date: 2026-09-19 (Asia/Tokyo)
- Host: macOS arm64
- Python: 3.12.14
- Default config: `configs/default.yaml`
- AI execution: disabled by default

`configs/colab.yaml` は別経路としてVLM/LLMを有効化し、公式GISデータとHugging Faceモデルのオンライン自動取得を指定します。

標準利用経路はGoogle Colabです。ローカル実行は公開GitHubリポジトリをcloneした利用者向けの任意経路として文書化しています（ADR-009）。

## End-to-end pipeline

Command:

```bash
.venv/bin/python scripts/run_pipeline.py --config configs/default.yaml
```

Result: completed. `artifacts/run_metadata.json` has `status: completed`. The run used the configured official archives from the checksum-verified local cache and regenerated preprocessing, features, models, predictions, maps, VLM status JSON, integrated JSON, and Markdown report.

Verified minimum artifacts:

- `artifacts/figures/land_price_map.png`
- `artifacts/figures/land_price_change_map.png`
- `artifacts/figures/prediction_map.png`
- `artifacts/figures/residual_map.png`
- `artifacts/predictions/predictions.parquet`
- `artifacts/metrics/metrics.json`
- `artifacts/vlm/map_analysis.json`
- `artifacts/reports/analysis.json`
- `artifacts/reports/report.md`

All were present and nonempty. `AnalysisReport` and `VLMAnalysis` Pydantic validation passed.

## Real data observed

- Land-price sites: 1,715 in each year from 2019 through 2024; 1,687 in 2025.
- Linked next-year test labels for 2025: 1,674.
- Population meshes in Osaka: 5,076.
- Distinct land-price positions processed for context: 8,592.
- Raw official ZIP download size: approximately 69 MiB.
- Input land-price CRS: EPSG:4612 for 2019–2023; EPSG:6668 for 2024–2025.
- Analysis CRS: EPSG:6674.

The 2010 population value is census-based (with the source's correction method); 2020/2025 values are projections from the 2017 dataset. They were not treated as observed population change.

## ML result

The model selected by 2024 validation MAE was Random Forest. The untouched 2025 target-year test produced:

| Metric | Value |
|---|---:|
| n | 1,674 |
| MAE | 0.008766 fraction = 0.877 percentage points |
| RMSE | 0.013801 fraction = 1.380 percentage points |
| R² | 0.8868 |
| Direction accuracy | 0.9092 |

These values describe the held-out public land-price sites under the configured temporal split. They are not evidence of causal effects, full-property-market accuracy, or performance in unknown municipalities.

On this macOS/Accelerate environment, scikit-learn emitted NumPy `matmul` RuntimeWarnings while fitting/predicting Ridge. Transformed inputs and predictions were explicitly checked as finite, Ridge was not selected, and the warning did not occur in the tree-model result. This may not reproduce on Colab Linux.

## Automated tests and checks

```text
pytest -q: 33 passed
ruff check .: passed
all 9 scripts --help: exit code 0
```

The tests cover required columns, missing/geographic CRS rejection, projected distance, inclusive 1 km station count, deterministic mesh-boundary handling, area-weighted land use, flood coverage unknown vs zero risk, annual-site linking, movement exclusion, missing-year leakage, target-feature leakage, temporal split, four model shapes, metrics, Pydantic output, provider failure, ZIP path traversal, config cycles, stale artifact hashes, and Notebook logic separation.

Local AI device selection is also covered: CUDA has priority, Apple MPS is selected when CUDA is absent, and CPU is not used silently. Notebook code may import only the read-only `osaka_geo_ai.presentation` adapter; configuration resolution and runtime-device display are implemented there. This host did not have the optional PyTorch AI environment installed, so real MPS model inference remains unverified.

## Notebook

更新後Notebookはnbformat、全コードセルのAST、Presentation Layer制約、GitHub clone、script呼出し、AI有効設定の契約テストを通過しました（24セル、うちコード12セル）。コードセルを同一Pythonプロセスで上から順に実行し、pipeline、設定・実行環境表、データ表、HTML地図、PNG、metrics、予測、VLM JSON、report-generation JSON、Markdownレポートの表示まで完了しました。Notebookからのdomain importは読み取り専用の `osaka_geo_ai.presentation` だけです。

Colab判定はロード済みモジュールに依存せず、`google.colab` の取得可否で行います。editable install直後のカーネルでもimportできるよう、cloneしたrepositoryの `src/` を明示的にimport pathへ登録します。ローカル環境ではセットアップからPresentation adapterのimportまでを新しいPythonプロセスで再実行し、成功を確認しました。

再実行時のGit更新は、浅いcloneへ `pull --ff-only` せず、浅くfetchした `FETCH_HEAD` を直接checkoutします。一時bare repositoryで旧commitを浅くcloneした後にremoteを更新する条件を再現し、checkout後のcommit SHAが最新remoteと一致することを確認しました。

A native Jupyter kernel run could not be performed in the managed sandbox because the kernel manager needs to bind a local communication port. An escalation request was not executed because automatic approval review hit the account usage limit; this was not a safety rejection. Actual Google Colab execution and CUDA inference therefore remain external verification items.

## Maps

The generated static maps were visually inspected. The four-panel comparison uses the same EPSG:6674 extent and includes titles, reference periods, units, direction, scale bar, color bars, and source/process attribution. The population panel explicitly labels itself as a 2017-vintage projection rather than observed change. The residual panel defines residual as actual minus predicted.

## VLM and LLM

- Default run status: `vlm.status = disabled`, with empty findings and an explicit limitation.
- Default report mode: deterministic template based on `analysis.json`.
- Mock provider tests verified successful structured output, invalid free prose, invalid confidence, unavailable provider, inference failure, and empty-findings behavior.
- Real Qwen2-VL/Qwen2.5 LLM model loading and inference were not executed because this host has no configured CUDA runtime and the optional AI packages were not installed.
- `configs/colab.yaml` is supplied for Colab GPU execution. It enables online Hugging Face downloads and pins model revisions.
- `python scripts/run_pipeline.py --config configs/colab.yaml` completed locally using the verified public-data cache. As designed for a non-CUDA host without AI extras, VLM/LLM recorded `unavailable` and the numerical pipeline plus deterministic report completed. Real model download/inference remains a Colab GPU verification item.
