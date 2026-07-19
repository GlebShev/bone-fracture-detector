from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image


@dataclass(frozen=True, slots=True)
class RawDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


class Detector(Protocol):
    def predict(self, image: Image.Image, confidence: float) -> list[RawDetection]: ...


class UltralyticsDetector:
    """Small adapter keeping Ultralytics imports out of API-only environments."""

    def __init__(self, weights_path: Path, image_size: int) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - exercised only in ML runtime
            raise RuntimeError(
                "Ultralytics could not be imported. Install requirements-ml.txt "
                f"and verify native OpenCV dependencies. Original error: {exc}"
            ) from exc

        self._model = YOLO(str(weights_path))
        self._image_size = image_size

    def predict(self, image: Image.Image, confidence: float) -> list[RawDetection]:
        results = self._model.predict(
            source=image,
            conf=confidence,
            imgsz=self._image_size,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names
        detections: list[RawDetection] = []
        for coordinates, score, class_value in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
            strict=True,
        ):
            class_id = int(class_value)
            class_name = (
                str(names.get(class_id, class_id))
                if isinstance(names, dict)
                else str(names[class_id])
            )
            detections.append(
                RawDetection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(score),
                    bbox=tuple(float(value) for value in coordinates),
                )
            )
        return detections
