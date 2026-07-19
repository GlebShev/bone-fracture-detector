from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

SLICED_FALLBACK_FRACTION = 0.60
SLICED_FALLBACK_MIN_CONFIDENCE = 0.28
SLICED_FALLBACK_NMS_IOU = 0.50
SLICED_FALLBACK_SUPPORT_CONFIDENCE = 0.03
SLICED_FALLBACK_SUPPORT_IOU = 0.50


@dataclass(frozen=True, slots=True)
class RawDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


class Detector(Protocol):
    def predict(
        self,
        image: Image.Image,
        confidence: float,
        sensitivity_mode: bool = False,
    ) -> list[RawDetection]: ...


class UltralyticsDetector:
    """Small adapter keeping Ultralytics imports out of API-only environments."""

    def __init__(
        self,
        weights_path: Path,
        image_size: int,
        *,
        sliced_fallback: bool = False,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - exercised only in ML runtime
            raise RuntimeError(
                "Ultralytics could not be imported. Install requirements-ml.txt "
                f"and verify native OpenCV dependencies. Original error: {exc}"
            ) from exc

        self._model = YOLO(str(weights_path))
        self._image_size = image_size
        self._sliced_fallback = sliced_fallback

    def predict(
        self,
        image: Image.Image,
        confidence: float,
        sensitivity_mode: bool = False,
    ) -> list[RawDetection]:
        fallback_enabled = self._sliced_fallback and sensitivity_mode
        initial_confidence = (
            min(confidence, SLICED_FALLBACK_SUPPORT_CONFIDENCE)
            if fallback_enabled
            else confidence
        )
        candidates = self._predict_images([image], initial_confidence)[0]
        detections = [item for item in candidates if item.confidence >= confidence]
        if detections or not fallback_enabled:
            return detections

        crops = overlapping_strips(image, SLICED_FALLBACK_FRACTION)
        fallback_confidence = max(confidence, SLICED_FALLBACK_MIN_CONFIDENCE)
        sliced_detections = self._predict_images(
            [crop.image for crop in crops],
            fallback_confidence,
            rect=False,
        )
        mapped = [
            offset_detection(detection, crop.offset_x, crop.offset_y)
            for crop, crop_detections in zip(crops, sliced_detections, strict=True)
            for detection in crop_detections
        ]
        deduplicated = non_max_suppression(mapped, SLICED_FALLBACK_NMS_IOU)
        return [
            detection
            for detection in deduplicated
            if any(
                detection.class_id == support.class_id
                and bbox_iou(detection, support) >= SLICED_FALLBACK_SUPPORT_IOU
                for support in candidates
            )
        ]

    def _predict_images(
        self,
        images: list[Image.Image],
        confidence: float,
        *,
        rect: bool = True,
    ) -> list[list[RawDetection]]:
        results = self._model.predict(
            source=images,
            conf=confidence,
            imgsz=self._image_size,
            verbose=False,
            rect=rect,
        )
        return [detections_from_result(result) for result in results]


@dataclass(frozen=True, slots=True)
class ImageCrop:
    image: Image.Image
    offset_x: int
    offset_y: int


def overlapping_strips(image: Image.Image, fraction: float) -> tuple[ImageCrop, ImageCrop]:
    """Return two overlapping strips along the image's shorter dimension."""
    if not 0.5 < fraction < 1.0:
        raise ValueError("fraction must be between 0.5 and 1.0")

    width, height = image.size
    if width >= height:
        strip_height = max(1, round(height * fraction))
        second_offset = height - strip_height
        return (
            ImageCrop(image.crop((0, 0, width, strip_height)), 0, 0),
            ImageCrop(
                image.crop((0, second_offset, width, height)),
                0,
                second_offset,
            ),
        )

    strip_width = max(1, round(width * fraction))
    second_offset = width - strip_width
    return (
        ImageCrop(image.crop((0, 0, strip_width, height)), 0, 0),
        ImageCrop(
            image.crop((second_offset, 0, width, height)),
            second_offset,
            0,
        ),
    )


def detections_from_result(result: object) -> list[RawDetection]:
    """Convert one Ultralytics result without leaking its types into the package."""
    boxes = result.boxes  # type: ignore[attr-defined]
    if boxes is None or len(boxes) == 0:
        return []

    names = result.names  # type: ignore[attr-defined]
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


def offset_detection(detection: RawDetection, offset_x: int, offset_y: int) -> RawDetection:
    x_min, y_min, x_max, y_max = detection.bbox
    return RawDetection(
        class_id=detection.class_id,
        class_name=detection.class_name,
        confidence=detection.confidence,
        bbox=(
            x_min + offset_x,
            y_min + offset_y,
            x_max + offset_x,
            y_max + offset_y,
        ),
    )


def bbox_iou(first: RawDetection, second: RawDetection) -> float:
    first_x_min, first_y_min, first_x_max, first_y_max = first.bbox
    second_x_min, second_y_min, second_x_max, second_y_max = second.bbox
    intersection_width = max(
        0.0,
        min(first_x_max, second_x_max) - max(first_x_min, second_x_min),
    )
    intersection_height = max(
        0.0,
        min(first_y_max, second_y_max) - max(first_y_min, second_y_min),
    )
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first_x_max - first_x_min) * max(0.0, first_y_max - first_y_min)
    second_area = max(0.0, second_x_max - second_x_min) * max(0.0, second_y_max - second_y_min)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def non_max_suppression(
    detections: list[RawDetection],
    iou_threshold: float,
) -> list[RawDetection]:
    kept: list[RawDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        overlaps_existing = any(
            detection.class_id == other.class_id
            and bbox_iou(detection, other) >= iou_threshold
            for other in kept
        )
        if not overlaps_existing:
            kept.append(detection)
    return kept
