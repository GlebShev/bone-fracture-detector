from __future__ import annotations

import threading
import time
from collections.abc import Callable

from PIL import Image

from fracture_detector.config import ModelSpec
from fracture_detector.detector import Detector, UltralyticsDetector
from fracture_detector.errors import ModelNotFoundError, ModelUnavailableError
from fracture_detector.imaging import annotate_image, image_to_base64
from fracture_detector.schemas import Detection, ModelInfo, PredictionResponse

DetectorFactory = Callable[[ModelSpec], Detector]


def _default_factory(spec: ModelSpec) -> Detector:
    return UltralyticsDetector(
        spec.weights_path,
        spec.image_size,
        sliced_fallback=spec.sliced_fallback,
    )


class ModelManager:
    """Lazy model registry. Each model is loaded once and protected during inference."""

    def __init__(
        self,
        specs: tuple[ModelSpec, ...],
        factory: DetectorFactory = _default_factory,
    ) -> None:
        self._specs = {spec.id: spec for spec in specs}
        self._factory = factory
        self._models: dict[str, Detector] = {}
        self._errors: dict[str, str] = {}
        self._load_lock = threading.Lock()
        self._inference_locks = {model_id: threading.Lock() for model_id in self._specs}

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id=spec.id,
                title=spec.title,
                description=spec.description,
                available=spec.weights_path.is_file() and spec.id not in self._errors,
                image_size=spec.image_size,
                error=self._errors.get(spec.id),
            )
            for spec in self._specs.values()
        ]

    def _get_model(self, model_name: str) -> Detector:
        spec = self._specs.get(model_name)
        if spec is None:
            raise ModelNotFoundError(f"Неизвестная модель: {model_name}.")
        if not spec.weights_path.is_file():
            raise ModelUnavailableError(
                f"Веса модели {model_name} не найдены: {spec.weights_path}."
            )
        if model_name in self._models:
            return self._models[model_name]

        with self._load_lock:
            if model_name in self._models:
                return self._models[model_name]
            try:
                model = self._factory(spec)
            except Exception as exc:
                self._errors[model_name] = str(exc)
                raise ModelUnavailableError(
                    f"Не удалось загрузить модель {model_name}: {exc}"
                ) from exc
            self._models[model_name] = model
            self._errors.pop(model_name, None)
            return model

    def predict(
        self,
        image: Image.Image,
        model_name: str,
        confidence: float,
        sensitivity_mode: bool = False,
    ) -> PredictionResponse:
        if not 0.05 <= confidence <= 0.95:
            raise ValueError("confidence должен находиться в диапазоне [0.05, 0.95].")

        model = self._get_model(model_name)
        effective_sensitivity_mode = (
            sensitivity_mode and self._specs[model_name].sliced_fallback
        )
        with self._inference_locks[model_name]:
            started = time.perf_counter()
            raw_detections = model.predict(
                image,
                confidence,
                effective_sensitivity_mode,
            )
            latency_ms = (time.perf_counter() - started) * 1000

        annotated = annotate_image(image, raw_detections)
        detections = [
            Detection(
                class_id=item.class_id,
                class_name=item.class_name,
                confidence=item.confidence,
                bbox=item.bbox,
            )
            for item in raw_detections
        ]
        return PredictionResponse(
            model_name=model_name,
            sensitivity_mode=effective_sensitivity_mode,
            latency_ms=round(latency_ms, 2),
            image_width=image.width,
            image_height=image.height,
            detection_count=len(detections),
            detections=detections,
            annotated_image_base64=image_to_base64(annotated),
        )
