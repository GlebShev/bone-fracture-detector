FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY requirements-api.txt ./

# OpenCV (imported by Ultralytics) needs these shared libraries in slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Render's free service is CPU-only. Installing PyTorch from its CPU index avoids
# several gigabytes of unused CUDA libraries and keeps the image deployable.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 torchvision==0.17.2
RUN pip install --no-cache-dir -r requirements-api.txt

COPY fracture_detector ./fracture_detector
COPY backend ./backend
COPY models ./models

EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
