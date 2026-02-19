# syntax=docker/dockerfile:1
# Backend production image for FastAPI (NavispaceAI)
FROM python:3.11-slim AS runtime

# Ensure predictable Python runtime behavior.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Optional ML dependency installation toggle (walls/doors segmentation model).
ARG INSTALL_ML=false

# System deps kept minimal for OpenCV/trimesh runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install optional ML dependencies only when requested.
COPY backend/ml/requirements-ml.txt /app/backend/ml/requirements-ml.txt
RUN if [ "$INSTALL_ML" = "true" ]; then \
      pip install --no-cache-dir -r /app/backend/ml/requirements-ml.txt ; \
    fi

# Copy project source.
COPY backend /app/backend
COPY assets /app/assets

# Runtime defaults (overridable via env).
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_RELOAD=false \
    BUILDING_ASSETS_PATH=/app/assets

EXPOSE 8000

# Start API server on configured port.
CMD ["sh", "-c", "uvicorn backend.main:app --host ${API_HOST} --port ${API_PORT}"]
