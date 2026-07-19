from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import create_app
from fracture_detector.config import ModelSpec, Settings
from fracture_detector.detector import RawDetection
from fracture_detector.model_manager import ModelManager


class FakeDetector:
    def predict(
        self,
        image: Image.Image,
        confidence: float,
        sensitivity_mode: bool = False,
    ) -> list[RawDetection]:
        if confidence > 0.9:
            return []
        return [
            RawDetection(
                class_id=2,
                class_name="forearm fracture",
                confidence=0.91,
                bbox=(10.0, 12.0, 50.0, 55.0),
            )
        ]


@pytest.fixture
def image_bytes() -> bytes:
    image = Image.new("RGB", (80, 64), color=(32, 36, 42))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def app_factory(tmp_path: Path) -> Callable[..., FastAPI]:
    def factory(*, weights_exist: bool = True) -> FastAPI:
        fast_weights = tmp_path / "fast.pt"
        accurate_weights = tmp_path / "accurate.pt"
        if weights_exist:
            fast_weights.touch()
            accurate_weights.touch()
        specs = (
            ModelSpec(
                "fast",
                "Fast",
                "fast model",
                fast_weights,
                640,
                sliced_fallback=True,
            ),
            ModelSpec("accurate", "Accurate", "accurate model", accurate_weights, 768),
        )
        settings = Settings(model_specs=specs, max_upload_bytes=1024 * 1024)
        manager = ModelManager(specs, factory=lambda _: FakeDetector())
        return create_app(settings=settings, model_manager=manager)

    return factory


@pytest.fixture
def client(app_factory: Callable[..., FastAPI]) -> TestClient:
    return TestClient(app_factory())
