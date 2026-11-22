# Construir desde la RAÍZ del repositorio:
#   docker build -f fase-4/Dockerfile -t titanic-monitoreo .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_VERSION=v1 \
    MODELS_DIR=/app/fase-2/models \
    DATA_DIR=/app/data \
    LOG_PATH=/app/fase-4/logs/registro_predicciones.csv

WORKDIR /app

COPY fase-4/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY fase-2/train.py fase-2/predict.py ./fase-2/
COPY fase-2/src ./fase-2/src
COPY fase-2/models ./fase-2/models
COPY data/raw ./data/raw
COPY fase-3/api.py fase-3/phase2_imports.py fase-3/training_service.py ./fase-3/
COPY fase-4/api.py fase-4/monitoreo.py ./fase-4/
COPY fase-4/logs ./fase-4/logs

WORKDIR /app/fase-4
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
