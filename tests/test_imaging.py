from __future__ import annotations

import io

import pytest
from PIL import Image

from fracture_detector.errors import InvalidImageError
from fracture_detector.imaging import decode_image


def make_image(width: int, height: int, image_format: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (width, height), color=128).save(buffer, format=image_format)
    return buffer.getvalue()


def test_decode_image_normalizes_to_rgb() -> None:
    image = decode_image(make_image(20, 10), max_pixels=1_000)
    assert image.mode == "RGB"
    assert image.size == (20, 10)


def test_decode_image_enforces_pixel_limit() -> None:
    with pytest.raises(InvalidImageError, match="размеры"):
        decode_image(make_image(20, 20), max_pixels=399)


def test_decode_image_rejects_empty_payload() -> None:
    with pytest.raises(InvalidImageError, match="пуст"):
        decode_image(b"", max_pixels=1_000)
