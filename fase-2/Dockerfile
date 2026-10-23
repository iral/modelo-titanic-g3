FROM python:3.13.0-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY train.py predict.py pytest.ini ./
COPY src ./src
COPY tests ./tests
COPY data/ejemplo_prediccion.csv ./data/ejemplo_prediccion.csv

CMD ["python", "train.py", "--help"]
