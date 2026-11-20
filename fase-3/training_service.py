"""Servicio de entrenamiento versionado de la Fase 3.

Reutiliza train_version de la Fase 2 y añade validación HTTP, asignación de
versiones y publicación atómica de los artefactos completos.
"""

import logging
import re
import shutil
import threading
import uuid
from pathlib import Path

import pandas as pd

from phase2_imports import train_version
from src.config import (
    DEFAULT_SEED,
    FEATURE_COLUMNS,
    REPOSITORY_ROOT,
    TARGET_DEFAULT,
)
from src.errors import DataValidationError

logger = logging.getLogger("titanic.train")

VERSION_PATTERN = re.compile(r"^v(\d+)$")
_TRAIN_LOCK = threading.Lock()


class DataFileNotFoundError(Exception):
    """El archivo de datos solicitado no existe."""


class InvalidTrainingConfigError(Exception):
    """Un parámetro de la solicitud no es válido."""


class InvalidTrainingDataError(Exception):
    """Los datos no permiten entrenar (columnas ausentes o valores inválidos)."""


class ArtifactSaveError(Exception):
    """No se pudieron guardar o publicar los artefactos."""


class TrainingFailedError(Exception):
    """Falló el entrenamiento por una causa interna inesperada."""


def next_version_id(models_dir: Path) -> str:
    """Devuelve el siguiente identificador libre (v1, v2, ...)."""
    numbers = []
    if models_dir.is_dir():
        for entry in models_dir.iterdir():
            match = VERSION_PATTERN.match(entry.name)
            if entry.is_dir() and match:
                numbers.append(int(match.group(1)))
    return f"v{max(numbers, default=0) + 1}"


def resolve_training_data(data_path, data_dir: Path) -> Path:
    """Resuelve el CSV predeterminado o valida una ruta explícita dentro de DATA_DIR."""
    allowed_dir = Path(data_dir).resolve()
    if data_path is None:
        candidate = allowed_dir / "raw" / "train.csv"
    else:
        candidate = Path(data_path)
        if not candidate.is_absolute():
            candidate = REPOSITORY_ROOT / candidate

    # resolve() sigue enlaces simbólicos; la comprobación posterior impide
    # escapar de DATA_DIR mediante rutas relativas o symlinks.
    resolved = candidate.resolve()
    if not resolved.is_relative_to(allowed_dir):
        raise InvalidTrainingConfigError(
            "El parámetro 'data_path' debe apuntar a un archivo dentro del "
            f"directorio de datos permitido ({allowed_dir})."
        )
    if not resolved.is_file():
        raise DataFileNotFoundError(
            f"No se encontró el archivo de datos configurado: {resolved.name}."
        )
    return resolved


def validate_training_columns(csv_path: Path, target: str) -> None:
    """Valida columnas y contenido binario del target antes de entrenar."""
    if target != TARGET_DEFAULT:
        raise InvalidTrainingConfigError(
            f"La variable objetivo permitida para este proyecto es '{TARGET_DEFAULT}'."
        )
    try:
        dataframe = pd.read_csv(csv_path)
    except Exception as exc:
        raise InvalidTrainingDataError(
            f"No se pudo leer '{csv_path.name}' como CSV."
        ) from exc

    missing = [c for c in [*FEATURE_COLUMNS, target] if c not in dataframe.columns]
    if missing:
        raise InvalidTrainingDataError(
            "El archivo de datos no contiene las columnas necesarias: "
            + ", ".join(missing)
            + "."
        )

    target_values = dataframe[target]
    if target_values.isna().any():
        raise InvalidTrainingDataError(
            f"La variable objetivo '{target}' contiene valores nulos."
        )
    observed = set(pd.to_numeric(target_values, errors="coerce").dropna().unique())
    if pd.api.types.is_bool_dtype(target_values.dtype) or not target_values.map(
        lambda value: value in (0, 1, 0.0, 1.0) and not isinstance(value, bool)
    ).all():
        raise InvalidTrainingDataError(
            f"La variable objetivo '{target}' solo puede contener 0 y 1."
        )
    if observed != {0, 1}:
        raise InvalidTrainingDataError(
            f"La variable objetivo '{target}' debe contener ambas clases: 0 y 1."
        )


def _publish_version(staged_dir: Path, final_dir: Path) -> None:
    """Publica la versión completa de una sola vez (cambio de nombre)."""
    if final_dir.exists():
        raise FileExistsError(f"La versión {final_dir.name} ya existe.")
    staged_dir.rename(final_dir)


def train_new_version(
    models_dir: Path,
    data_dir: Path,
    data_path=None,
    target=None,
    seed=None,
) -> dict:
    """Entrena con la Fase 2 y publica una versión nueva sin tocar las existentes."""
    target = TARGET_DEFAULT if target is None else target
    seed = DEFAULT_SEED if seed is None else seed

    if target != TARGET_DEFAULT:
        raise InvalidTrainingConfigError(
            f"La variable objetivo permitida para este proyecto es '{TARGET_DEFAULT}'."
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 2**32 - 1:
        raise InvalidTrainingConfigError("'seed' debe ser un entero entre 0 y 4294967295.")

    csv_path = resolve_training_data(data_path, data_dir)
    validate_training_columns(csv_path, target)

    with _TRAIN_LOCK:
        staging_root = models_dir / f".tmp-{uuid.uuid4().hex}"
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
            version = next_version_id(models_dir)
            try:
                result = train_version(
                    data_path=csv_path,
                    target=target,
                    output_dir=staging_root,
                    version=version,
                    seed=seed,
                )
            except DataValidationError as exc:
                raise InvalidTrainingDataError(str(exc)) from exc
            except (ValueError, TypeError) as exc:
                raise InvalidTrainingDataError(
                    "Los datos no cumplen los requisitos del entrenamiento."
                ) from exc
            except OSError as exc:
                logger.exception("Error de archivo durante el entrenamiento")
                raise ArtifactSaveError(
                    "No se pudieron leer o guardar los archivos del entrenamiento."
                ) from exc
            except Exception as exc:
                logger.exception("Falló el entrenamiento de la versión %s", version)
                raise TrainingFailedError("Falló el entrenamiento del modelo.") from exc

            try:
                _publish_version(Path(result["version_dir"]), models_dir / version)
            except OSError as exc:
                logger.exception("No se pudo publicar la versión %s", version)
                raise ArtifactSaveError(
                    "No se pudo publicar la versión entrenada."
                ) from exc
        except OSError as exc:
            logger.exception("No se pudo preparar el directorio de modelos")
            raise ArtifactSaveError(
                "No se pudo preparar el directorio de modelos."
            ) from exc
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    return {
        "version": version,
        "variable_objetivo": target,
        "semilla": seed,
        "fecha_creacion": result["metadata"]["fecha_creacion"],
        "metricas": result["metrics"],
    }
