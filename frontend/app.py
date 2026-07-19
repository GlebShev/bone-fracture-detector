from __future__ import annotations

import base64
import io
import os
from typing import Any

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(
    page_title="Детектор переломов",
    page_icon="🩻",
    layout="wide",
)

MODELS_CONNECT_TIMEOUT_SECONDS = 10
MODELS_READ_TIMEOUT_SECONDS = 90
PREDICT_CONNECT_TIMEOUT_SECONDS = 10
PREDICT_READ_TIMEOUT_SECONDS = 300


def get_api_url() -> str:
    try:
        secret_url = st.secrets.get("API_URL", "")
    except StreamlitSecretNotFoundError:
        secret_url = ""
    return str(secret_url or os.getenv("API_URL", "http://localhost:8000")).rstrip("/")


@st.cache_data(ttl=30, show_spinner=False)
def fetch_models(api_url: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{api_url}/models",
        timeout=(MODELS_CONNECT_TIMEOUT_SECONDS, MODELS_READ_TIMEOUT_SECONDS),
    )
    response.raise_for_status()
    return response.json()["models"]


def show_error(response: requests.Response) -> None:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    st.error(f"Ошибка сервиса: {detail}")


api_url = get_api_url()

st.title("Детекция переломов на рентгеновских снимках")
st.caption("Загрузите снимок, выберите модель и порог уверенности.")

try:
    with st.spinner("Подключаемся к backend…"):
        model_options = fetch_models(api_url)
except requests.RequestException:
    model_options = [
        {
            "id": "fast",
            "title": "Fast",
            "description": "Быстрая модель",
            "available": False,
        },
        {
            "id": "accurate",
            "title": "Accurate",
            "description": "YOLO11s, 768 px",
            "available": False,
        },
    ]
    st.warning("Backend пока недоступен. Проверьте API_URL и состояние сервиса.")

labels = {
    model["id"]: f"{model['title']} — {model['description']}"
    + ("" if model.get("available") else " (веса недоступны)")
    for model in model_options
}

with st.sidebar:
    st.header("Настройки")
    model_name = st.selectbox(
        "Модель",
        options=list(labels),
        format_func=labels.get,
    )
    confidence = st.slider(
        "Минимальная уверенность",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Чем ниже порог, тем больше детекций.",
    )
    sensitivity_mode = st.checkbox(
        "Повышенная чувствительность",
        value=model_name == "fast",
        disabled=model_name != "fast",
        key=f"sensitivity_mode_{model_name}",
        help=(
            "Если обычный Fast-проход пуст, модель проверяет два перекрывающихся "
            "фрагмента. Режим повышает recall и число false positive."
        ),
    )
    if sensitivity_mode and model_name == "fast":
        st.caption("При пустом результате запускаются два дополнительных прохода.")
    st.divider()
    st.markdown(f"Backend: `{api_url}`")

uploaded_file = st.file_uploader(
    "Загрузите рентгеновский снимок",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened_image:
            source_image = opened_image.convert("RGB")
    except (OSError, UnidentifiedImageError):
        st.error("Не удалось открыть изображение.")
        st.stop()
    source_column, result_column = st.columns(2)
    with source_column:
        st.subheader("Исходное изображение")
        st.image(source_image, use_container_width=True)

    if st.button("Запустить детекцию", type="primary", use_container_width=True):
        with st.spinner("Модель анализирует снимок…"):
            try:
                response = requests.post(
                    f"{api_url}/predict",
                    files={
                        "file": (
                            uploaded_file.name,
                            image_bytes,
                            uploaded_file.type or "image/jpeg",
                        )
                    },
                    data={
                        "model_name": model_name,
                        "confidence": confidence,
                        "sensitivity_mode": sensitivity_mode and model_name == "fast",
                    },
                    timeout=(
                        PREDICT_CONNECT_TIMEOUT_SECONDS,
                        PREDICT_READ_TIMEOUT_SECONDS,
                    ),
                )
            except requests.RequestException as exc:
                st.error(f"Не удалось связаться с backend: {exc}")
            else:
                if not response.ok:
                    show_error(response)
                else:
                    result = response.json()
                    annotated_bytes = base64.b64decode(result["annotated_image_base64"])
                    with result_column:
                        st.subheader("Результат")
                        st.image(annotated_bytes, use_container_width=True)

                    metric_1, metric_2, metric_3 = st.columns(3)
                    metric_1.metric("Найдено областей", result["detection_count"])
                    metric_2.metric("Инференс", f"{result['latency_ms']:.1f} мс")
                    metric_3.metric("Модель", result["model_name"].title())

                    if result["detections"]:
                        rows = [
                            {
                                "Класс": detection["class_name"],
                                "Уверенность": round(detection["confidence"], 3),
                                "Bounding box": ", ".join(
                                    f"{coordinate:.0f}" for coordinate in detection["bbox"]
                                ),
                            }
                            for detection in result["detections"]
                        ]
                        st.dataframe(rows, use_container_width=True, hide_index=True)
                    else:
                        st.info(
                            "При выбранном пороге детекций нет. Попробуйте снизить "
                            "confidence или включить повышенную чувствительность."
                        )
else:
    st.info("Поддерживаются изображения JPEG, PNG и WEBP размером до 10 МБ.")
