from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from fracture_detector.config import Settings
from fracture_detector.errors import (
    InvalidImageError,
    ModelNotFoundError,
    ModelUnavailableError,
)
from fracture_detector.imaging import decode_image
from fracture_detector.model_manager import ModelManager
from fracture_detector.schemas import (
    HealthResponse,
    ModelsResponse,
    PredictionResponse,
)


def create_app(
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    manager = model_manager or ModelManager(settings.model_specs)

    app = FastAPI(
        title="Bone Fracture Detection API",
        description=(
            "Учебный API локализации возможных переломов. "
            "Результаты не являются медицинским заключением."
        ),
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        available_count = sum(model.available for model in manager.list_models())
        return HealthResponse(
            ready=available_count > 0,
            available_models=available_count,
        )

    @app.get("/models", response_model=ModelsResponse, tags=["models"])
    def models() -> ModelsResponse:
        return ModelsResponse(models=manager.list_models())

    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    async def predict(
        file: Annotated[
            UploadFile,
            File(description="JPEG, PNG or WEBP X-ray image"),
        ],
        model_name: Annotated[str, Form()] = "fast",
        confidence: Annotated[float, Form()] = 0.25,
    ) -> PredictionResponse:
        if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=415, detail="Неподдерживаемый тип файла.")

        payload = await file.read(settings.max_upload_bytes + 1)
        if len(payload) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Изображение слишком большое.")

        try:
            image = decode_image(payload, settings.max_image_pixels)
            return await run_in_threadpool(
                manager.predict,
                image,
                model_name,
                confidence,
            )
        except InvalidImageError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
