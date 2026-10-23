"""Entrenamiento versionado del modelo de supervivencia del Titanic.

Uso:
    python train.py --data ../data/raw/train.csv --target Survived \
        --output-dir models --version v1 --seed 42
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier

from src.artifacts import (
    ensure_version_available,
    save_version,
    version_directory,
)
from src.config import (
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL_PARAMS,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PRIMARY_METRIC,
    TARGET_DEFAULT,
)
from src.data import (
    load_csv,
    prepare_training_data,
    split_dataset,
)
from src.errors import AppError
from src.metrics import evaluate_estimator
from src.pipeline import build_pipeline


def _git_output(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _git_commit() -> str:
    return _git_output("rev-parse", "HEAD") or "desconocido"


def _resolve_author(author: str | None) -> str:
    if author:
        return author
    return _git_output("config", "user.name") or "equipo"


def _feature_schema() -> list:
    return [
        {
            "nombre": column,
            "tipo": "numerica" if column in NUMERIC_FEATURES else "categorica",
        }
        for column in FEATURE_COLUMNS
    ]


def train_version(
    data_path,
    target: str = TARGET_DEFAULT,
    output_dir="models",
    version: str = "v1",
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    model_params: dict | None = None,
    author: str | None = None,
) -> dict:
    """Entrena una versión completa y devuelve los artefactos generados."""
    if not 0 < test_size < 1:
        raise AppError("El tamaño del conjunto de prueba debe estar entre 0 y 1.")

    output_dir = Path(output_dir)
    ensure_version_available(output_dir, version)

    dataframe = load_csv(data_path)
    features, target_series = prepare_training_data(dataframe, target)
    features_train, features_test, target_train, target_test = split_dataset(
        features, target_series, test_size, seed
    )

    pipeline = build_pipeline(seed=seed, params=model_params)
    pipeline.fit(features_train, target_train)
    model_metrics = evaluate_estimator(pipeline, features_test, target_test)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(features_train, target_train)
    baseline_metrics = evaluate_estimator(baseline, features_test, target_test)

    metrics = {
        "modelo": model_metrics,
        "modelo_base": baseline_metrics,
        "metrica_principal": PRIMARY_METRIC,
    }

    metadata = {
        "version": version,
        "fecha_creacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tipo_problema": "clasificacion_binaria",
        "variable_objetivo": target,
        "algoritmo": "RandomForestClassifier",
        "variables_entrada": _feature_schema(),
        "hiperparametros": {
            **DEFAULT_MODEL_PARAMS,
            **(model_params or {}),
            "random_state": seed,
        },
        "semilla": seed,
        "datos": {
            "fuente": "Kaggle - Titanic: Machine Learning from Disaster",
            "archivo": str(data_path),
            "filas": int(len(dataframe)),
            "entrenamiento": int(len(features_train)),
            "prueba": int(len(features_test)),
        },
        "separacion": {
            "estrategia": "train_test_split estratificado",
            "test_size": float(test_size),
            "random_state": seed,
            "estratificado": True,
        },
        "metrica_principal": PRIMARY_METRIC,
        "metricas": metrics,
        "python": platform.python_version(),
        "librerias": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "dependencias": "requirements.txt",
        "commit_codigo": _git_commit(),
        "autor": _resolve_author(author),
    }

    version_dir = version_directory(output_dir, version)
    paths = save_version(version_dir, pipeline, metrics, metadata)

    return {
        "version": version,
        "version_dir": version_dir,
        "paths": paths,
        "metrics": metrics,
        "metadata": metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena el modelo de supervivencia del Titanic y guarda una versión "
            "con su pipeline, métricas y metadatos."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Ruta del CSV etiquetado (debe incluir la variable objetivo).",
    )
    parser.add_argument(
        "--target",
        default=TARGET_DEFAULT,
        help="Nombre de la variable objetivo.",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Carpeta donde se crean las versiones.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Identificador único de la versión, por ejemplo v1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Semilla aleatoria.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Proporción del conjunto de prueba.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Número de árboles (por defecto 100).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Profundidad máxima (por defecto 5).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=None,
        help="Mínimo de muestras por hoja (por defecto 2).",
    )
    parser.add_argument(
        "--author",
        default=None,
        help="Autor de la ejecución (por defecto, el usuario de git).",
    )
    return parser


def _model_params_from_args(args) -> dict:
    params = {}
    if args.n_estimators is not None:
        params["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        params["max_depth"] = args.max_depth
    if args.min_samples_leaf is not None:
        params["min_samples_leaf"] = args.min_samples_leaf
    return params


def _print_summary(result: dict) -> None:
    metrics = result["metrics"]
    model = metrics["modelo"]
    baseline = metrics["modelo_base"]
    print(f"Versión entrenada: {result['version']}")
    print(f"  Modelo : {result['paths']['modelo']}")
    print(f"  Métricas: {result['paths']['metricas']}")
    print(f"  Metadatos: {result['paths']['metadatos']}")
    print(
        "  RandomForest  "
        f"accuracy={model['accuracy']:.4f}  f1={model['f1']:.4f}  "
        f"roc_auc={model['roc_auc']}"
    )
    print(
        "  Baseline      "
        f"accuracy={baseline['accuracy']:.4f}  f1={baseline['f1']:.4f}  "
        f"roc_auc={baseline['roc_auc']}"
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = train_version(
            data_path=args.data,
            target=args.target,
            output_dir=args.output_dir,
            version=args.version,
            seed=args.seed,
            test_size=args.test_size,
            model_params=_model_params_from_args(args),
            author=args.author,
        )
    except AppError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error de archivo: {exc}", file=sys.stderr)
        return 1

    _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
