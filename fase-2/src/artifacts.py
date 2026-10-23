"""Guardado y carga de versiones del modelo."""

import json
from pathlib import Path

import joblib

from .errors import (
    ModelLoadError,
    ModelNotFoundError,
    VersionAlreadyExistsError,
)

MODEL_FILENAME = "modelo.joblib"
METRICS_FILENAME = "metrics.json"
METADATA_FILENAME = "metadata.json"


def version_directory(output_dir, version: str) -> Path:
    """Devuelve la carpeta de una versión concreta."""
    return Path(output_dir) / version


def ensure_version_available(output_dir, version: str) -> Path:
    """Impide sobrescribir una versión existente."""
    directory = version_directory(output_dir, version)
    if directory.exists():
        raise VersionAlreadyExistsError(
            f"La versión '{version}' ya existe en {directory}. "
            "Selecciona otro identificador; no se sobrescribirá."
        )
    return directory


def resolve_model_path(model_path) -> Path:
    """Acepta la ruta del artefacto o la carpeta de la versión."""
    path = Path(model_path)
    if path.is_dir():
        path = path / MODEL_FILENAME
    if not path.is_file():
        raise ModelNotFoundError(f"No se encontró el artefacto del modelo: {path}")
    return path


def load_model(model_path):
    """Carga el pipeline guardado sin reentrenar ni reajustar transformaciones."""
    path = resolve_model_path(model_path)
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(
            f"No se pudo cargar el modelo {path}: {exc}"
        ) from exc


def save_json(path, payload) -> None:
    """Escribe un JSON legible y reproducible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_json(path):
    """Lee un JSON de la versión del modelo."""
    path = Path(path)
    if not path.is_file():
        raise ModelNotFoundError(f"No se encontró el archivo: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"El archivo {path} no es un JSON válido: {exc}") from exc


def save_version(directory, model, metrics: dict, metadata: dict) -> dict:
    """Serializa el pipeline, las métricas y los metadatos de una versión."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)

    model_path = directory / MODEL_FILENAME
    metrics_path = directory / METRICS_FILENAME
    metadata_path = directory / METADATA_FILENAME

    joblib.dump(model, model_path)
    save_json(metrics_path, metrics)
    save_json(metadata_path, metadata)

    return {
        "modelo": model_path,
        "metricas": metrics_path,
        "metadatos": metadata_path,
    }
