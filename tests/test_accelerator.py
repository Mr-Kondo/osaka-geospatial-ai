from types import SimpleNamespace

import pytest

from osaka_geo_ai.accelerator import inference_dtype, resolve_device


class DeviceBackend:
    def __init__(self, available):
        self.available = available
        self.cleared = False

    def is_available(self):
        return self.available

    def empty_cache(self):
        self.cleared = True


def fake_torch(*, cuda=False, mps=False):
    return SimpleNamespace(
        cuda=DeviceBackend(cuda),
        mps=DeviceBackend(mps),
        backends=SimpleNamespace(mps=DeviceBackend(mps)),
        float16="float16",
        float32="float32",
    )


def test_auto_device_prefers_cuda_then_apple_mps():
    assert resolve_device(fake_torch(cuda=True, mps=True), "auto") == "cuda"
    assert resolve_device(fake_torch(mps=True), "auto") == "mps"


def test_auto_device_falls_back_to_cpu():
    assert resolve_device(fake_torch(), "auto") == "cpu"


def test_mps_uses_half_precision_and_rejects_unavailable_backend():
    torch = fake_torch(mps=True)
    assert inference_dtype(torch, "mps") == "float16"
    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device(fake_torch(), "mps")
