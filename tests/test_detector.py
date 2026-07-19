from __future__ import annotations

from PIL import Image

from fracture_detector.detector import (
    RawDetection,
    UltralyticsDetector,
    bbox_iou,
    non_max_suppression,
    offset_detection,
    overlapping_strips,
)


class TensorLike:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def cpu(self) -> TensorLike:
        return self

    def tolist(self) -> list[object]:
        return self._values


class Boxes:
    def __init__(self, coordinates: list[list[float]], confidences: list[float]) -> None:
        self.xyxy = TensorLike(coordinates)
        self.conf = TensorLike(confidences)
        self.cls = TensorLike([0.0] * len(confidences))

    def __len__(self) -> int:
        return len(self.conf._values)


class Result:
    names = {0: "fracture"}

    def __init__(self, boxes: Boxes) -> None:
        self.boxes = boxes


class FallbackModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[Result]:
        self.calls.append(kwargs)
        sources = kwargs["source"]
        assert isinstance(sources, list)
        if len(sources) == 1:
            return [Result(Boxes([[1.0, 22.0, 11.0, 32.0]], [0.04]))]
        return [
            Result(Boxes([], [])),
            Result(Boxes([[1.0, 2.0, 11.0, 12.0]], [0.4])),
        ]


def detection(confidence: float, bbox: tuple[float, float, float, float]) -> RawDetection:
    return RawDetection(0, "fracture", confidence, bbox)


def test_overlapping_strips_follow_short_dimension() -> None:
    landscape = overlapping_strips(Image.new("RGB", (100, 50)), 0.6)
    assert [crop.image.size for crop in landscape] == [(100, 30), (100, 30)]
    assert [(crop.offset_x, crop.offset_y) for crop in landscape] == [(0, 0), (0, 20)]

    portrait = overlapping_strips(Image.new("RGB", (50, 100)), 0.6)
    assert [crop.image.size for crop in portrait] == [(30, 100), (30, 100)]
    assert [(crop.offset_x, crop.offset_y) for crop in portrait] == [(0, 0), (20, 0)]


def test_offset_detection_maps_crop_coordinates_to_original() -> None:
    source = detection(0.75, (1.0, 2.0, 11.0, 12.0))
    mapped = offset_detection(source, 20, 30)
    assert mapped.bbox == (21.0, 32.0, 31.0, 42.0)
    assert mapped.confidence == source.confidence


def test_non_max_suppression_keeps_best_overlapping_box() -> None:
    best = detection(0.9, (0.0, 0.0, 10.0, 10.0))
    duplicate = detection(0.8, (1.0, 1.0, 11.0, 11.0))
    separate = detection(0.7, (20.0, 20.0, 30.0, 30.0))

    assert bbox_iou(best, duplicate) > 0.5
    assert non_max_suppression([duplicate, separate, best], 0.5) == [best, separate]


def test_ultralytics_adapter_uses_square_padded_fallback_and_maps_box() -> None:
    detector = UltralyticsDetector.__new__(UltralyticsDetector)
    detector._model = FallbackModel()
    detector._image_size = 640
    detector._sliced_fallback = True

    result = detector.predict(
        Image.new("RGB", (100, 50)),
        confidence=0.25,
        sensitivity_mode=True,
    )

    assert result == [detection(0.4, (1.0, 22.0, 11.0, 32.0))]
    assert detector._model.calls[0]["rect"] is True
    assert detector._model.calls[1]["rect"] is False
    assert detector._model.calls[1]["conf"] == 0.28
