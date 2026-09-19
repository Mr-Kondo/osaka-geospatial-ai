"""Generate a factual fallback or a model explanation using analysis.json only."""

from __future__ import annotations

import gc
import json
import logging
import os
from typing import Protocol

import requests

from osaka_geo_ai.config import artifact
from osaka_geo_ai.io import read_json, sha256, write_json
from osaka_geo_ai.llm.integration import integrate
from osaka_geo_ai.llm.prompts import SYSTEM_PROMPT
from osaka_geo_ai.llm.schemas import AnalysisReport
from osaka_geo_ai.vision.vlm import ProviderUnavailable

LOG = logging.getLogger(__name__)


class ReportProvider(Protocol):
    def generate(self, analysis: dict) -> str: ...


class HuggingFaceReporter:
    def __init__(self, settings):
        self.settings = settings

    def generate(self, analysis):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ProviderUnavailable(
                "Install optional AI dependencies: pip install -e '.[ai]'"
            ) from exc
        settings = self.settings
        device = settings["device"]
        if device == "cuda" and not torch.cuda.is_available():
            raise ProviderUnavailable("LLM: CUDA GPU unavailable")
        model = None
        try:
            torch.manual_seed(settings.get("seed", 42))
            LOG.info(
                "Downloading/loading Hugging Face LLM %s at revision %s",
                settings["model"],
                settings["revision"],
            )
            tokenizer = AutoTokenizer.from_pretrained(
                settings["model"],
                revision=settings["revision"],
                local_files_only=settings.get("local_files_only", False),
                trust_remote_code=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                settings["model"],
                revision=settings["revision"],
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                local_files_only=settings.get("local_files_only", False),
                trust_remote_code=False,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(analysis, ensure_ascii=False)},
            ]
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
            if tokens.shape[-1] + settings["max_new_tokens"] > model.config.max_position_embeddings:
                raise ProviderUnavailable(
                    "analysis.json exceeds this model's context limit; choose a larger context model"
                )
            with torch.inference_mode():
                output = model.generate(
                    tokens, max_new_tokens=settings["max_new_tokens"], do_sample=False
                )
            return tokenizer.decode(output[0, tokens.shape[-1] :], skip_special_tokens=True)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class OpenAICompatibleReporter:
    def __init__(self, settings):
        self.settings = settings

    def generate(self, analysis):
        settings = self.settings
        token = os.environ.get(settings["api_key_env"])
        if not token or not settings["base_url"]:
            raise ProviderUnavailable(
                "Set llm.base_url and the configured API-key environment variable"
            )
        response = requests.post(
            settings["base_url"].rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": settings["model"],
                "temperature": 0,
                "max_tokens": settings["max_new_tokens"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(analysis, ensure_ascii=False)},
                ],
            },
            timeout=settings["timeout_seconds"],
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def format_number(value, decimals=2):
    return "欠測" if value is None else f"{value:,.{decimals}f}"


def template_report(analysis):
    observed, model = analysis["observed_statistics"], analysis["model"]
    metrics = model["metrics"]
    errors = analysis["residual_analysis"]["municipalities"][:5]
    vlm = analysis["vlm_analysis"]
    worst = "\n".join(
        f"| {x['municipality_name']} | {x['n']} | {x['mae'] * 100:.2f} | {x['mean_residual'] * 100:+.2f} |"
        for x in errors
    )
    visuals = "VLMは実行されていないか失敗しており、モデルによる視覚的観察結果はありません。"
    if vlm["status"] == "completed":
        visuals = (
            "以下はVLMによる視覚的観察であり、数値的な検証や因果関係の証拠ではありません。\n\n"
            + "\n".join(
                f"- {p['region']}: {p['observation']}（モデル自己評価 {p['confidence']:.2f}）"
                for p in vlm["visual_patterns"]
            )
        )
    change = observed["land_price_change_rate"]["mean"]
    sources = "\n".join(
        f"- [{s['dataset']}]({s['url']}) — {s['license']}" for s in analysis["sources"]
    )
    limitations = "\n".join(f"- {s}" for s in analysis["limitations"])
    forecast = analysis["prediction_summary"]["forecast"]
    return f"""# 大阪府 地価・地域変化分析

> 数値はPython/GIS/MLで算出したものです。この本文は検証済みJSONから生成した定型説明で、LLMの生成文章ではありません。

## 概要

大阪府の{observed["year"]}年地価公示を対象に、{observed["sample_sites"]:,}地点を分析しました。地価公示は標準地の評価額であり、売買価格や全土地の相場そのものではありません。

## 現在確認できる傾向

**観測された事実:** 対象地点の公示価格の中央値は{format_number(observed["land_price_jpy_per_m2"]["median"], 0)}円/㎡です。

**統計的分析結果:** 前年と接続できた{observed["land_price_change_rate"]["n"]:,}地点の平均変化率は{format_number(None if change is None else change * 100)}%でした。地点ごとの単純平均であり、府全体の土地面積に応じた平均ではありません。

## 地理的特徴

人口メッシュとの対応率は{analysis["spatial_analysis"]["population_mesh_match_fraction"] * 100:.1f}%です。駅までの直線距離の中央値は{format_number(analysis["spatial_analysis"]["station_distance_m"]["median"], 0)}mでした。徒歩距離ではありません。

人口図は2010年を基準とする2017年作成の将来推計を示します。2020年・2025年に実測された人口とは区別してください。行政区域別集計・相関係数の詳細は `analysis.json` に収録しています。相関から地価変動の原因は確定できません。

## 地価予測

**MLによる予測:** 翌年の地価変化率を予測します。検証年のMAEで `{model["name"]}` を選び、{model["test_target_year"]}年の未学習データで評価しました。

| テスト指標 | 値 | 意味 |
|---|---:|---|
| MAE | {metrics["mae"] * 100:.2f} pp | 誤差の絶対値の平均 |
| RMSE | {metrics["rmse"] * 100:.2f} pp | 大きな誤差を重く評価 |
| R² | {format_number(metrics["r2"], 3)} | 平均値予測と比べた適合の指標。負にもなる |
| 方向一致率 | {metrics["direction_accuracy"] * 100:.1f}% | 上昇・横ばい・下落の符号が一致した割合 |

`pp` はパーセントポイントです。実際が5%、予測が3%なら残差は+2 ppです。

別の `forecast.parquet` は{forecast["origin_year"]}年時点を想定した{forecast["target_year"]}年予測です。実行日からの将来予測ではなく、この設定では実現値を評価していません。

## モデルが予測しにくかった地域

**統計的分析結果:** 対象地点のMAEが大きかった市区町村を示します。地点数の少ない地域の順位は不安定です。

| 市区町村 | 地点数 | MAE (pp) | 平均残差 (pp) |
|---|---:|---:|---:|
{worst}

残差は実際の変化率から予測を引いた値です。正は過小予測、負は過大予測です。残差の集中だけで、その原因を特定することはできません。

## 地図から確認できる視覚的特徴

**VLMによる視覚的観察:** status=`{vlm["status"]}`。

{visuals}

## 一般向けの説明

**解釈:** わかりやすく言うと、過去の地価と地域の特徴から翌年の変化を予測し、予測が外れた場所を地図で確認する仕組みです。予測の誤差と入力データの古さを踏まえて読む必要があります。

## 注意点・不確実性

{limitations}

## 出典

国土交通省の以下のデータを本PoCが編集・加工しています。モデル予測とレポートは国の公式見解ではありません。

{sources}
"""


def generate_report(config, provider: ReportProvider | None = None):
    path = integrate(config)
    analysis = AnalysisReport.model_validate(read_json(path)).model_dump()
    text = template_report(analysis)
    settings = config["llm"]
    status = {
        "status": "disabled",
        "provider": settings["provider"],
        "model": settings["model"],
        "mode": "deterministic_template",
        "analysis_sha256": sha256(path),
        "error": None,
    }
    generated_path = artifact(config, "reports/llm_explanation.md")
    generated_path.unlink(missing_ok=True)
    if settings["enabled"]:
        try:
            if provider is None:
                providers = {
                    "huggingface": HuggingFaceReporter,
                    "openai_compatible": OpenAICompatibleReporter,
                }
                if settings["provider"] not in providers:
                    raise ProviderUnavailable(f"Unsupported LLM provider: {settings['provider']}")
                provider = providers[settings["provider"]](
                    {**settings, "seed": config["project"]["seed"]}
                )
            generated = provider.generate(analysis).strip()
            if not generated or "## 注意点・不確実性" not in generated:
                raise ValueError(
                    "LLM response is empty/truncated or missing the uncertainty section"
                )
            generated_path.write_text(generated + "\n", encoding="utf-8")
            text += (
                "\n---\n\n## LLMによる説明（自動生成・内容未検証）\n\n以下は上記のJSONを入力したLLMの説明です。数値と区分の照合には本文とanalysis.jsonを用いてください。\n\n"
                + generated
                + "\n"
            )
            status.update(status="completed", mode="template_plus_llm")
        except ProviderUnavailable as exc:
            status.update(status="unavailable", error=str(exc))
            LOG.warning("LLM unavailable; retaining factual template: %s", exc)
        except Exception as exc:
            status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            LOG.warning("LLM failed; retaining factual template: %s", exc)
    output = artifact(config, "reports/report.md")
    output.write_text(text, encoding="utf-8")
    write_json(artifact(config, "reports/report_generation.json"), status)
    return output
