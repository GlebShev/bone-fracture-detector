from __future__ import annotations

import base64
import io

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from fracture_detector.detector import RawDetection
from fracture_detector.errors import InvalidImageError

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
PALETTE = (
    "#00A6FB",
    "#FF7A00",
    "#2DC653",
    "#9D4EDD",
    "#F72585",
    "#FFD60A",
    "#00B4D8",
)


def decode_image(payload: bytes, max_pixels: int) -> Image.Image:
    if not payload:
        raise InvalidImageError("Файл пуст.")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format not in SUPPORTED_FORMATS:
                raise InvalidImageError("Поддерживаются только JPEG, PNG и WEBP.")
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise InvalidImageError("Изображение имеет недопустимые размеры.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
            return image
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("Не удалось прочитать изображение.") from exc


def annotate_image(image: Image.Image, detections: list[RawDetection]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    width, height = annotated.size

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        box = (
            max(0.0, min(x1, width - 1)),
            max(0.0, min(y1, height - 1)),
            max(0.0, min(x2, width - 1)),
            max(0.0, min(y2, height - 1)),
        )
        color = PALETTE[detection.class_id % len(PALETTE)]
        draw.rectangle(box, outline=color, width=max(2, round(min(width, height) / 250)))
        label = f"{detection.class_name} {detection.confidence:.2f}"
        label_box = draw.textbbox((box[0], box[1]), label, font=font, stroke_width=1)
        text_height = label_box[3] - label_box[1] + 6
        text_width = label_box[2] - label_box[0] + 8
        text_y = max(0.0, box[1] - text_height)
        draw.rectangle(
            (box[0], text_y, min(width - 1, box[0] + text_width), text_y + text_height),
            fill=color,
        )
        draw.text(
            (box[0] + 4, text_y + 3),
            label,
            fill="black",
            font=font,
            stroke_width=0,
        )
    return annotated


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
