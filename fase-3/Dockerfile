# Construir desde la RAÍZ del repositorio:
#   docker build -f fase-3/Dockerfile -t titanic-api .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_VERSION=v1 \
    MODELS_DIR=/app/fase-2/models \
    DATA_DIR=/app/data

WORKDIR /app

# uv dentro de la imagen; las versiones exactas vienen de requirements.txt.
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv
COPY fase-3/requirements.txt ./requirements.txt
RUN uv pip install --system --break-system-packages -r requirements.txt

COPY fase-2/train.py fase-2/predict.py ./fase-2/
COPY fase-2/src ./fase-2/src
COPY fase-2/models ./fase-2/models
COPY fase-2/data/ejemplo_prediccion.csv ./fase-2/data/ejemplo_prediccion.csv
COPY data/raw ./data/raw
COPY fase-3/api.py fase-3/phase2_imports.py fase-3/training_service.py ./fase-3/
COPY fase-3/tests ./fase-3/tests

WORKDIR /app/fase-3
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
