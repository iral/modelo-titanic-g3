"""Entrenamiento versionado del modelo de supervivencia del Titanic.

Uso:
    python train.py --target Survived --output-dir models --version v1 --seed 42

Por defecto usa el dataset del repositorio (data/raw/train.csv); --data permite
indicar otro archivo.
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
    DEFAULT_CV_FOLDS,
    DEFAULT_DATA_PATH,
    DEFAULT_GROUP_COLUMN,
    DEFAULT_MODEL_PARAMS,
    DEFAULT_SEED,
    DEFAULT_SELECT_BY,
    DEFAULT_TEST_SIZE,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PRIMARY_METRIC,
    REPOSITORY_ROOT,
    TARGET_DEFAULT,
)
from src.data import (
    build_groups,
    count_shared_groups,
    load_csv,
    prepare_training_data,
    split_dataset,
)
from src.errors import AppError
from src.metrics import evaluate_estimator
from src.pipeline import build_pipeline, build_search


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


def _resolve_data_path(data_path) -> Path:
    """Resuelve rutas relativas contra la raíz del repositorio cuando hace falta."""
    path = Path(data_path)
    if not path.is_absolute() and not path.exists():
        candidate = REPOSITORY_ROOT / path
        if candidate.exists():
            return candidate
    return path


def _portable_path(path) -> str:
    """Guarda rutas del repositorio de forma relativa y portable."""
    try:
        return Path(path).resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def _feature_schema() -> list:
    return [
        {
            "nombre": column,
            "tipo": "numerica" if column in NUMERIC_FEATURES else "categorica",
        }
        for column in FEATURE_COLUMNS
    ]


def train_version(
    data_path=None,
    target: str = TARGET_DEFAULT,
    output_dir="models",
    version: str = "v1",
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    model_params: dict | None = None,
    author: str | None = None,
    select_by: str = DEFAULT_SELECT_BY,
    cv_folds: int = DEFAULT_CV_FOLDS,
    group_column: str | None = DEFAULT_GROUP_COLUMN,
    param_grid: dict | None = None,
) -> dict:
    """Entrena una versión completa y devuelve los artefactos generados."""
    if not 0 < test_size < 1:
        raise AppError("El tamaño del conjunto de prueba debe estar entre 0 y 1.")
    if select_by not in {"f1", "accuracy", "recall", "precision"}:
        raise AppError(
            "La métrica de selección debe ser f1, accuracy, recall o precision."
        )
    if cv_folds < 2:
        raise AppError("La validación cruzada necesita al menos 2 particiones.")

    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    resolved_data_path = _resolve_data_path(data_path)

    output_dir = Path(output_dir)
    ensure_version_available(output_dir, version)

    dataframe = load_csv(resolved_data_path)
    groups = build_groups(dataframe, group_column)
    features, target_series = prepare_training_data(dataframe, target)
    features_train, features_test, target_train, target_test = split_dataset(
        features, target_series, test_size, seed, groups=groups
    )
    groups_train = groups.loc[features_train.index] if groups is not None else None
    groups_test = groups.loc[features_test.index] if groups is not None else None

    if model_params:
        pipeline = build_pipeline(seed=seed, params=model_params)
        pipeline.fit(features_train, target_train)
        best_params = {**DEFAULT_MODEL_PARAMS, **model_params}
        selection = {
            "metodo": "configuracion_fija",
            "metrica": select_by,
            "configuracion": model_params,
        }
    else:
        search = build_search(
            build_pipeline(seed=seed),
            groups_train,
            seed=seed,
            select_by=select_by,
            cv_folds=cv_folds,
            param_grid=param_grid,
        )
        fit_params = (
            {"groups": groups_train.to_numpy()} if groups_train is not None else {}
        )
        search.fit(features_train, target_train, **fit_params)
        pipeline = search.best_estimator_
        best_params = {
            key.replace("classifier__", ""): value
            for key, value in search.best_params_.items()
        }
        selection = {
            "metodo": "GridSearchCV",
            "metrica": select_by,
            "folds": cv_folds,
            "mejor_score_cv": float(search.best_score_),
            "combinaciones": len(search.cv_results_["params"]),
        }

    model_metrics = evaluate_estimator(pipeline, features_test, target_test)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(features_train, target_train)
    baseline_metrics = evaluate_estimator(baseline, features_test, target_test)

    metrics = {
        "modelo": model_metrics,
        "modelo_base": baseline_metrics,
        "metrica_principal": PRIMARY_METRIC,
    }

    descriptors = (
        {
            "columna": group_column,
            "grupos_entrenamiento": int(groups_train.nunique()),
            "grupos_prueba": int(groups_test.nunique()),
            "grupos_compartidos": count_shared_groups(groups_train, groups_test),
        }
        if groups is not None
        else {"columna": None}
    )

    metadata = {
        "version": version,
        "fecha_creacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tipo_problema": "clasificacion_binaria",
        "variable_objetivo": target,
        "algoritmo": "RandomForestClassifier",
        "variables_entrada": _feature_schema(),
        "hiperparametros": {**best_params, "random_state": seed},
        "seleccion": selection,
        "semilla": seed,
        "datos": {
            "fuente": "Kaggle - Titanic: Machine Learning from Disaster",
            "archivo": _portable_path(resolved_data_path),
            "filas": int(len(dataframe)),
            "entrenamiento": int(len(features_train)),
            "prueba": int(len(features_test)),
        },
        "separacion": {
            "estrategia": (
                "StratifiedGroupKFold por ticket"
                if groups is not None
                else "train_test_split estratificado"
            ),
            "test_size": float(test_size),
            "random_state": seed,
            "estratificado": True,
            "por_grupos": groups is not None,
        },
        "grupos": descriptors,
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
        default=DEFAULT_DATA_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        help=(
            "Ruta del CSV etiquetado (debe incluir la variable objetivo). "
            "Las rutas relativas se resuelven contra la raíz del repositorio."
        ),
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
        help="Fija el número de árboles y omite la búsqueda de hiperparámetros.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Fija la profundidad máxima y omite la búsqueda de hiperparámetros.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=None,
        help="Fija el mínimo de muestras por hoja y omite la búsqueda.",
    )
    parser.add_argument(
        "--select-by",
        choices=["f1", "accuracy", "recall", "precision"],
        default=DEFAULT_SELECT_BY,
        help="Métrica usada para elegir hiperparámetros en validación cruzada.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help="Número de particiones de validación cruzada.",
    )
    parser.add_argument(
        "--group-column",
        default=DEFAULT_GROUP_COLUMN,
        help=(
            "Columna que define los grupos de viaje (por defecto Ticket). "
            "Usa una cadena vacía para desactivar la partición por grupos."
        ),
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
    metadata = result["metadata"]
    selection = metadata["seleccion"]
    groups = metadata["grupos"]
    print(f"Versión entrenada: {result['version']}")
    print(f"  Modelo : {result['paths']['modelo']}")
    print(f"  Métricas: {result['paths']['metricas']}")
    print(f"  Metadatos: {result['paths']['metadatos']}")
    if selection["metodo"] == "GridSearchCV":
        print(
            f"  Selección: GridSearchCV por {selection['metrica']} "
            f"(folds={selection['folds']}, combinaciones={selection['combinaciones']}, "
            f"mejor_cv={selection['mejor_score_cv']:.4f})"
        )
    else:
        print(f"  Selección: configuración fija ({selection['metrica']})")
    if groups.get("columna"):
        print(
            f"  Grupos ({groups['columna']}): "
            f"entrenamiento={groups['grupos_entrenamiento']}  "
            f"prueba={groups['grupos_prueba']}  "
            f"compartidos={groups['grupos_compartidos']}"
        )
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
            select_by=args.select_by,
            cv_folds=args.cv_folds,
            group_column=args.group_column,
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
