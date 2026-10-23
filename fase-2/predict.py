"""Predicción con una versión guardada del modelo.

Uso:
    python predict.py --model models/v1/modelo.joblib \
        --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import load_model, resolve_model_path
from src.config import (
    FEATURE_COLUMNS,
    OUTPUT_PREDICTION_COLUMN,
    OUTPUT_PROBABILITY_COLUMN,
)
from src.data import coerce_feature_types, load_csv
from src.errors import AppError


def _model_feature_columns(model) -> list:
    """Recupera el esquema de entrada con el que se entrenó el pipeline."""
    stored = getattr(model, "feature_names_in_", None)
    if stored is None:
        return list(FEATURE_COLUMNS)
    return [str(column) for column in stored]


def run_prediction(model_path, input_path, output_path) -> dict:
    """Genera una predicción por fila conservando el orden de la entrada."""
    model_file = resolve_model_path(model_path)
    model = load_model(model_file)
    dataframe = load_csv(input_path)

    feature_columns = _model_feature_columns(model)
    features = coerce_feature_types(dataframe, feature_columns)
    ignored_columns = [
        column for column in dataframe.columns if column not in feature_columns
    ]

    predictions = np.asarray(model.predict(features)).astype(int)

    output = pd.DataFrame({OUTPUT_PREDICTION_COLUMN: predictions})
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features))[:, 1]
        output[OUTPUT_PROBABILITY_COLUMN] = np.round(probabilities, 6)
    if "PassengerId" in dataframe.columns:
        output.insert(0, "PassengerId", dataframe["PassengerId"].to_numpy())

    if len(output) != len(dataframe):
        raise AppError(
            "La cantidad de predicciones no coincide con la cantidad de filas de entrada."
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_name(output_file.name + ".tmp")
    output.to_csv(temporary_file, index=False)
    os.replace(temporary_file, output_file)

    return {
        "predicciones": output,
        "filas": int(len(output)),
        "columnas_usadas": feature_columns,
        "columnas_ignoradas": ignored_columns,
        "salida": output_file,
        "modelo": model_file,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Carga una versión guardada del modelo y predice la supervivencia "
            "de los pasajeros de un CSV sin la variable objetivo."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ruta del artefacto modelo.joblib o de la carpeta de la versión.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV de entrada; no requiere la variable objetivo.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta del CSV de salida.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_prediction(args.model, args.input, args.output)
    except AppError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error de archivo: {exc}", file=sys.stderr)
        return 1

    if result["columnas_ignoradas"]:
        print(
            "Columnas ignoradas (no forman parte del esquema del modelo): "
            + ", ".join(result["columnas_ignoradas"])
        )
    print(f"Modelo utilizado: {result['modelo']}")
    print(f"Filas procesadas: {result['filas']}")
    print(f"Archivo generado: {result['salida']}")
    print(
        f"Supervivientes predichos: "
        f"{int(result['predicciones'][OUTPUT_PREDICTION_COLUMN].sum())}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
