"""Reduce numeric GIS/ML artifacts into a bounded, source-attributed LLM interface."""

from __future__ import annotations

from datetime import datetime, timezone

from osaka_geo_ai.config import artifact, directory
from osaka_geo_ai.data.loader import read_geo
from osaka_geo_ai.io import clean_json, read_json, sha256, write_json
from osaka_geo_ai.llm.schemas import AnalysisReport, VLMAnalysis
from osaka_geo_ai.models.predict import validate_prediction_contract


def summarize(values):
    values = values.dropna()
    return {
        "n": len(values),
        "mean": values.mean(),
        "median": values.median(),
        "min": values.min(),
        "max": values.max(),
        "p10": values.quantile(0.1),
        "p90": values.quantile(0.9),
    }


def integrate(config):
    validate_prediction_contract(config)
    features_path = directory(config, "processed") / "features.parquet"
    features = read_geo(features_path)
    metrics = read_json(artifact(config, "metrics/metrics.json"))
    if metrics["features_sha256"] != sha256(features_path):
        raise ValueError("Metrics refer to different features; retrain and regenerate predictions")
    map_manifest = read_json(artifact(config, "maps/map_manifest.json"))
    if map_manifest["features_sha256"] != sha256(features_path) or map_manifest[
        "prediction_metadata_sha256"
    ] != sha256(artifact(config, "predictions/prediction_metadata.json")):
        raise ValueError(
            "Maps refer to earlier predictions/features; rerun build_maps.py and analyze_map_vlm.py"
        )
    predictions = read_geo(artifact(config, "predictions/predictions.parquet"))
    forecast = read_geo(artifact(config, "predictions/forecast.parquet"))
    vlm = VLMAnalysis.model_validate(read_json(artifact(config, "vlm/map_analysis.json")))
    for item in vlm.input_images:
        if sha256(artifact(config, "figures/" + item["file"])) != item["sha256"]:
            raise ValueError("VLM inputs changed; rerun analyze_map_vlm.py")
    latest = features[features.year.eq(config["analysis"]["target_year"])]
    municipalities = (
        latest.groupby(["municipality_code", "municipality_name"])
        .agg(
            n=("site_id", "size"),
            median_land_price=("land_price", "median"),
            mean_change_rate=("land_price_change_rate", "mean"),
        )
        .reset_index()
    )
    errors = (
        predictions.groupby(["municipality_code", "municipality_name"])
        .agg(
            n=("site_id", "size"),
            mean_residual=("residual", "mean"),
            mae=("residual", lambda s: s.abs().mean()),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )
    worst = (
        predictions.assign(abs_residual=predictions.residual.abs())
        .sort_values("abs_residual", ascending=False)
        .head(10)
    )
    correlations = []
    for col in ["nearest_station_distance_m", "population_500m_2010", "built_up_ratio"]:
        pair = latest[["land_price", col]].dropna()
        value = pair.corr(method="spearman").iloc[0, 1] if len(pair) > 2 else None
        correlations.append(
            {
                "feature": col,
                "method": "Spearman rank correlation",
                "with": "latest official land price",
                "n": len(pair),
                "coefficient": value,
                "interpretation": "descriptive association, not causal evidence",
            }
        )
    limitations = [
        "地価公示の標準地が対象です。全土地の価格・実際の売買価格・府全体の面積加重平均を表しません。",
        "人口の基準値は補正を含む2010年国勢調査ベース、2020・2025年値は2017年作成の将来推計です。現在人口や観測された人口増減ではありません。",
        "鉄道は2020年、土地利用は2016年の固定スナップショットです。その後の新駅・土地利用変更を表しません。",
        "補助データは設定の利用可能年より前へ使用しません。ただし当時配布された全ファイルの履歴保存はなく、厳密な過去時点データの再現性は保証しません。",
        "前年番号・選定区分・用途・50m以内の位置を照合できる連続地点のみを教師ラベルにします。脱落・新設地点による選択バイアスがありえます。",
        "時系列検証であり、未知の市町村への空間外挿検証ではありません。短い学習期間で、景気・政策・感染症などをモデル化していません。",
        "MAE/RMSEは変化率のfraction単位です。100倍するとパーセントポイントです。予測区間や確率校正は未実装です。",
        "VLMの観察は仮説です。因果関係や正確な数値を保証しません。異なる年次の地図を比較しています。",
        "最新設定年からの予測は当該年時点の想定です。実行日の将来を意味せず、翌年の実現値として扱えません。",
    ]
    if not config["datasets"]["flood"]["enabled"]:
        limitations.append(
            "洪水データは未入力で、災害リスクは評価していません。欠測を安全と解釈できません。"
        )
    if vlm.status != "completed":
        limitations.append(f"VLM status={vlm.status}: モデルによる視覚分析結果はありません。")
    result = {
        "metadata": {
            "region": config["project"]["region"],
            "analysis_year": config["analysis"]["target_year"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "analysis_crs": config["analysis"]["analysis_crs"],
            "output_crs": config["analysis"]["output_crs"],
            "seed": config["project"]["seed"],
            "change_rate_unit": "fraction",
            "features_sha256": sha256(features_path),
            "evidence_classes": [
                "observed",
                "computed_statistics",
                "ml_prediction",
                "vlm_observation",
                "interpretation",
                "uncertainty",
            ],
        },
        "dataset_summary": read_json(artifact(config, "dataset_summary.json")),
        "observed_statistics": {
            "year": int(latest.year.iloc[0]),
            "sample_sites": len(latest),
            "land_price_jpy_per_m2": summarize(latest.land_price),
            "land_price_change_rate": summarize(latest.land_price_change_rate),
        },
        "spatial_analysis": {
            "municipality_statistics": municipalities.to_dict("records"),
            "descriptive_correlations": correlations,
            "population_mesh_match_fraction": float(latest.mesh_id.notna().mean()),
            "station_distance_m": summarize(latest.nearest_station_distance_m),
            "landuse_coverage_ratio": summarize(latest.landuse_coverage_ratio)
            if "landuse_coverage_ratio" in latest
            else None,
            "flood_status": "provided; consult coverage and scenario"
            if config["datasets"]["flood"]["enabled"]
            else "not_evaluated",
        },
        "model": {
            "name": metrics["selected_model"],
            "target": metrics["target"],
            "test_target_year": metrics["test_target_year"],
            "metrics": metrics["selected_test_metrics"],
            "selection": metrics["selection"],
            "validation_metrics": metrics["validation"],
            "test_comparison": metrics["test"],
        },
        "prediction_summary": {
            "held_out_test": summarize(predictions.prediction),
            "forecast": {
                "origin_year": int(forecast.year.iloc[0]),
                "target_year": int(forecast.target_year.iloc[0]),
                "fit_max_target_year": int(forecast.fit_max_target_year.iloc[0]),
                "prediction": summarize(forecast.prediction),
                "actual_available": False,
            },
        },
        "residual_analysis": {
            "definition": "actual minus predicted",
            "unit": "fraction",
            "summary": summarize(predictions.residual),
            "municipalities": errors.to_dict("records"),
            "largest_absolute_errors": worst[
                ["site_id", "municipality_name", "actual", "prediction", "residual"]
            ].to_dict("records"),
        },
        "vlm_analysis": vlm.model_dump(),
        "limitations": limitations,
        "sources": [
            {"dataset": name, "url": ds["source"], "license": ds["license"]}
            for name, ds in config["datasets"].items()
            if ds.get("enabled")
        ],
    }
    validated = AnalysisReport.model_validate(clean_json(result))
    output = artifact(config, "reports/analysis.json")
    write_json(output, validated.model_dump())
    return output
