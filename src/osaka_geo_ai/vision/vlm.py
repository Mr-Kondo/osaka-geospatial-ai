"""Provider abstraction with lazy OSS model loading and explicit degradation."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Protocol

from osaka_geo_ai.config import artifact
from osaka_geo_ai.io import read_json, sha256, write_json
from osaka_geo_ai.llm.schemas import VisualFindings, VLMAnalysis
from osaka_geo_ai.vision.prompts import map_prompt

LOG = logging.getLogger(__name__)


class ProviderUnavailable(RuntimeError):
    pass


class VLMProvider(Protocol):
    def analyze(self, images: list[Path], prompt: str) -> str: ...


def parse_findings(text: str) -> VisualFindings:
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    elif value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    return VisualFindings.model_validate(json.loads(value))


class HuggingFaceVLM:
    def __init__(self, settings):
        self.settings = settings

    def analyze(self, images, prompt):
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ProviderUnavailable(
                "Install optional AI dependencies: pip install -e '.[ai]'"
            ) from exc
        settings = self.settings
        device = settings["device"]
        if device == "cuda":
            if not torch.cuda.is_available():
                raise ProviderUnavailable("CUDA GPU unavailable; select a Colab GPU runtime")
            free, _ = torch.cuda.mem_get_info()
            if free / 1024**3 < settings["min_free_gpu_gb"]:
                raise ProviderUnavailable(
                    "Insufficient free GPU memory; lower image max_pixels or choose a smaller model"
                )
        torch.manual_seed(settings.get("seed", 42))
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = None
        try:
            LOG.info(
                "Downloading/loading Hugging Face VLM %s at revision %s",
                settings["model"],
                settings["revision"],
            )
            model = AutoModelForImageTextToText.from_pretrained(
                settings["model"],
                revision=settings["revision"],
                torch_dtype=dtype,
                device_map=device,
                local_files_only=settings.get("local_files_only", False),
                trust_remote_code=False,
            )
            processor = AutoProcessor.from_pretrained(
                settings["model"],
                revision=settings["revision"],
                max_pixels=settings["max_pixels"],
                local_files_only=settings.get("local_files_only", False),
                trust_remote_code=False,
            )
            content = []
            pixels = []
            for path in images:
                with Image.open(path) as im:
                    pixels.append(im.convert("RGB"))
                content.extend([{"type": "text", "text": path.name}, {"type": "image"}])
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=pixels, padding=True, return_tensors="pt").to(
                device
            )
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=settings["max_new_tokens"], do_sample=False
                )
            response = processor.batch_decode(
                output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            del inputs, output
            return response
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def analyze_maps(config, provider: VLMProvider | None = None):
    settings = config["vlm"]
    images = [artifact(config, f"figures/{name}") for name in settings["images"]]
    for path in images:
        if not path.exists():
            raise FileNotFoundError(f"Missing VLM input: {path}; run build_maps.py")
    payload = {
        "status": "disabled",
        "provider": settings["provider"],
        "model": settings["model"],
        "input_images": [{"file": p.name, "sha256": sha256(p)} for p in images],
        "visual_patterns": [],
        "anomalies": [],
        "cross_map_relationships": [],
        "limitations": ["VLMは未実行です。視覚的な観察結果は生成していません。"],
        "error": None,
    }
    raw_path = artifact(config, "vlm/raw_response.txt")
    raw_path.unlink(missing_ok=True)
    if settings["enabled"]:
        try:
            if provider is None:
                if settings["provider"] != "huggingface":
                    raise ProviderUnavailable(f"Unsupported VLM provider: {settings['provider']}")
                provider = HuggingFaceVLM({**settings, "seed": config["project"]["seed"]})
            LOG.info("Analyzing static maps with %s", settings["model"])
            text = provider.analyze(
                images, map_prompt(read_json(artifact(config, "maps/map_manifest.json")))
            )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(text, encoding="utf-8")
            findings = parse_findings(text)
            if any(set(r.maps) - set(settings["images"]) for r in findings.cross_map_relationships):
                raise ValueError("VLM referenced a map outside its input")
            payload.update(findings.model_dump())
            payload["status"] = "completed"
        except ProviderUnavailable as exc:
            payload.update(
                status="unavailable",
                error=str(exc),
                limitations=["VLMを実行できませんでした。", str(exc)],
            )
            LOG.warning("VLM unavailable: %s", exc)
        except Exception as exc:
            # AI failures are isolated; invalid JSON and OOM never become fabricated findings.
            reason = f"{type(exc).__name__}: {exc}"
            payload.update(
                status="failed",
                error=reason,
                limitations=["VLMのモデル読込・推論または構造検証に失敗しました。", reason],
            )
            LOG.warning("VLM failed: %s", reason)
    result = VLMAnalysis.model_validate(payload)
    path = artifact(config, "vlm/map_analysis.json")
    write_json(path, result.model_dump())
    return path
