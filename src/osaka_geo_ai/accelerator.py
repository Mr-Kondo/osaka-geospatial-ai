"""Resolve PyTorch inference devices without importing torch at module load time."""

from __future__ import annotations


def mps_available(torch) -> bool:
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(torch, requested: str, *, allow_cpu: bool = False) -> str:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_available(torch):
            return "mps"
        if allow_cpu:
            return "cpu"
        raise RuntimeError("No CUDA or Apple MPS accelerator is available")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA accelerator is unavailable")
    if requested == "mps" and not mps_available(torch):
        raise RuntimeError("Apple MPS accelerator is unavailable")
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported PyTorch device: {requested}")
    return requested


def inference_dtype(torch, device: str):
    return torch.float16 if device in {"cuda", "mps"} else torch.float32


def clear_device_cache(torch, device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and mps_available(torch):
        torch.mps.empty_cache()
