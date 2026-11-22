"""API de la Fase 4: Fase 3 + registro persistente de cada POST /predict.

No reimplementa la API: carga la aplicación de ``fase-3/api.py`` y añade un
middleware que escribe una fila por intento, incluidas las solicitudes
rechazadas por la validación de FastAPI (HTTP 422).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request, Response

BASE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = BASE_DIR.parent
FASE3_DIR = REPOSITORY_ROOT / "fase-3"
FASE2_DIR = REPOSITORY_ROOT / "fase-2"

for _path in (FASE3_DIR, FASE2_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load_phase3_app():
    """Carga fase-3/api.py sin colisionar con el módulo ``api`` de fase-4."""
    spec = importlib.util.spec_from_file_location(
        "fase3_api", FASE3_DIR / "api.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensivo
        raise RuntimeError("No se pudo cargar la API de la Fase 3.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_app()


app = _load_phase3_app()

LOG_HEADERS = ["fecha_hora", "estado_http", "version", "prediccion"]
_WRITE_LOCK = threading.Lock()


def default_log_path() -> Path:
    """Ruta del registro; se puede cambiar con la variable LOG_PATH."""
    configured = os.getenv("LOG_PATH")
    if configured:
        return Path(configured).expanduser()
    return BASE_DIR / "logs" / "registro_predicciones.csv"


def registrar_solicitud(estado_http: int, version: str = "", prediccion="") -> None:
    """Añade una fila al CSV, creando el encabezado solo la primera vez."""
    path = default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(LOG_HEADERS)
            writer.writerow(
                [
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    int(estado_http),
                    version,
                    prediccion,
                ]
            )


@app.middleware("http")
async def registrar_predicciones(request: Request, call_next):
    """Registra exactamente una fila por cada POST /predict."""
    response = await call_next(request)
    if request.method != "POST" or request.url.path != "/predict":
        return response

    body = b"".join([chunk async for chunk in response.body_iterator])
    version = ""
    prediccion = ""
    if response.status_code == 200:
        try:
            payload = json.loads(body)
            version = str(payload.get("version", ""))
            prediccion = payload.get("prediccion", "")
        except (TypeError, ValueError):
            pass

    registrar_solicitud(response.status_code, version, prediccion)

    # El cuerpo se consumió para leer la predicción: se reconstruye la respuesta
    # sin Content-Length para que Starlette calcule la longitud correcta.
    headers = {
        key: value
        for key, value in response.headers.items()
        if key.lower() != "content-length"
    }
    return Response(
        content=body,
        status_code=response.status_code,
        headers=headers,
        media_type=response.media_type,
    )
