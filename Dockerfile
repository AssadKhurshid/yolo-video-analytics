FROM python:3.11-slim

# OpenCV (pulled in by ultralytics) needs these system libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch: the default wheel bundles CUDA and is several GB larger.
# Installed first so this slow layer stays cached when only our code changes.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[api,video]"

# Download the default model at build time, so the container works offline.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

EXPOSE 8000
CMD ["uvicorn", "yva.api:app", "--host", "0.0.0.0", "--port", "8000"]
