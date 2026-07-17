from __future__ import annotations

import base64
import io
from collections.abc import Callable

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image


def test_health_and_model_catalog(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "ready": True, "available_models": 2}

    models = client.get("/models")
    assert models.status_code == 200
    assert [model["id"] for model in models.json()["models"]] == ["fast", "accurate"]
    assert all(model["available"] for model in models.json()["models"])


def test_predict_returns_detection_and_annotated_png(
    client: TestClient, image_bytes: bytes
) -> None:
    response = client.post(
        "/predict",
        files={"file": ("xray.png", image_bytes, "image/png")},
        data={"model_name": "accurate", "confidence": "0.25"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "accurate"
    assert body["detection_count"] == 1
    assert body["detections"][0]["class_name"] == "forearm fracture"
    assert body["detections"][0]["confidence"] == 0.91

    annotated = Image.open(io.BytesIO(base64.b64decode(body["annotated_image_base64"])))
    assert annotated.format == "PNG"
    assert annotated.size == (80, 64)


def test_high_threshold_can_return_empty_result(
    client: TestClient, image_bytes: bytes
) -> None:
    response = client.post(
        "/predict",
        files={"file": ("xray.png", image_bytes, "image/png")},
        data={"model_name": "fast", "confidence": "0.95"},
    )
    assert response.status_code == 200
    assert response.json()["detection_count"] == 0


def test_rejects_invalid_payload(client: TestClient) -> None:
    wrong_type = client.post(
        "/predict",
        files={"file": ("note.txt", b"not an image", "text/plain")},
    )
    assert wrong_type.status_code == 415

    fake_png = client.post(
        "/predict",
        files={"file": ("fake.png", b"not an image", "image/png")},
    )
    assert fake_png.status_code == 422


def test_rejects_unknown_model(client: TestClient, image_bytes: bytes) -> None:
    response = client.post(
        "/predict",
        files={"file": ("xray.png", image_bytes, "image/png")},
        data={"model_name": "unknown", "confidence": "0.25"},
    )
    assert response.status_code == 404


def test_reports_missing_weights(
    app_factory: Callable[..., FastAPI], image_bytes: bytes
) -> None:
    client = TestClient(app_factory(weights_exist=False))
    assert client.get("/health").json()["ready"] is False
    response = client.post(
        "/predict",
        files={"file": ("xray.png", image_bytes, "image/png")},
        data={"model_name": "fast", "confidence": "0.25"},
    )
    assert response.status_code == 503
