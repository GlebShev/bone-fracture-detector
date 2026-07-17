from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    ready: bool
    available_models: int


class ModelInfo(BaseModel):
    id: str
    title: str
    description: str
    available: bool
    image_size: int
    error: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]


class PredictionResponse(BaseModel):
    model_name: str
    latency_ms: float
    image_width: int
    image_height: int
    detection_count: int
    detections: list[Detection]
    annotated_image_base64: str
    annotated_image_media_type: str = "image/png"
