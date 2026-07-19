from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    title: str
    description: str
    weights_path: Path
    image_size: int
    sliced_fallback: bool = False


@dataclass(frozen=True, slots=True)
class Settings:
    model_specs: tuple[ModelSpec, ...]
    cors_origins: tuple[str, ...] = ("http://localhost:8501",)
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 40_000_000

    @classmethod
    def from_env(cls) -> Settings:
        model_dir = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models")))
        fast_path = Path(os.getenv("FAST_MODEL_PATH", str(model_dir / "fast.pt")))
        accurate_path = Path(
            os.getenv("ACCURATE_MODEL_PATH", str(model_dir / "accurate.pt"))
        )
        origins = tuple(
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
            if origin.strip()
        )
        max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "10"))
        return cls(
            model_specs=(
                ModelSpec(
                    id="fast",
                    title="Fast",
                    description="Компактная YOLO11n, 640 px: минимальная задержка.",
                    weights_path=fast_path,
                    image_size=640,
                    sliced_fallback=True,
                ),
                ModelSpec(
                    id="accurate",
                    title="Accurate",
                    description="YOLO11s, 768 px: выше детализация, больше задержка.",
                    weights_path=accurate_path,
                    image_size=768,
                ),
            ),
            cors_origins=origins,
            max_upload_bytes=max_upload_mb * 1024 * 1024,
        )
