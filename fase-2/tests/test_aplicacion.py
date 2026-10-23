"""Pruebas de la aplicación de la Fase 2.

Todas las pruebas usan datos sintéticos pequeños y directorios temporales
(tmp_path): no leen archivos de la entrega ni requieren conexión de red.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predict import run_prediction
from src.data import coerce_feature_types
from src.errors import DataValidationError, VersionAlreadyExistsError
from train import train_version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICT_SCRIPT = PROJECT_ROOT / "predict.py"
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"


def make_dataset(rows: int = 120, seed: int = 7) -> pd.DataFrame:
    """Genera un conjunto sintético con el esquema del Titanic y algunos nulos."""
    rng = np.random.default_rng(seed)
    dataframe = pd.DataFrame(
        {
            "PassengerId": np.arange(1, rows + 1),
            "Pclass": rng.integers(1, 4, rows),
            "Name": [f"Pasajero {index}" for index in range(rows)],
            "Sex": rng.choice(["male", "female"], rows),
            "Age": np.round(rng.normal(30, 10, rows), 1),
            "SibSp": rng.integers(0, 3, rows),
            "Parch": rng.integers(0, 3, rows),
            "Ticket": [f"T{index}" for index in range(rows)],
            "Fare": np.round(rng.uniform(5, 200, rows), 2),
            "Cabin": rng.choice([None, "C1", "B2"], rows),
            "Embarked": rng.choice(["S", "C", "Q"], rows),
            "Survived": rng.integers(0, 2, rows),
        }
    )
    dataframe.loc[0:4, "Age"] = np.nan
    dataframe.loc[0:2, "Embarked"] = np.nan
    return dataframe


@pytest.fixture()
def dataset_path(tmp_path: Path) -> Path:
    path = tmp_path / "train.csv"
    make_dataset().to_csv(path, index=False)
    return path


@pytest.fixture()
def trained_version(tmp_path: Path, dataset_path: Path) -> dict:
    return train_version(
        data_path=dataset_path,
        output_dir=tmp_path / "models",
        version="v1",
        seed=42,
    )


def run_cli(arguments: list, cwd: Path) -> subprocess.CompletedProcess:
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        arguments,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )


def test_entrenamiento_genera_artefactos(trained_version, tmp_path):
    version_dir = trained_version["version_dir"]
    for filename in ("modelo.joblib", "metrics.json", "metadata.json"):
        assert (version_dir / filename).is_file()

    metrics = trained_version["metrics"]
    assert set(metrics["modelo"]) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }
    assert set(metrics["modelo_base"]) == set(metrics["modelo"])
    assert metrics["metrica_principal"] == "f1"

    metadata = trained_version["metadata"]
    assert metadata["version"] == "v1"
    assert metadata["variable_objetivo"] == "Survived"
    assert metadata["algoritmo"] == "RandomForestClassifier"
    assert metadata["semilla"] == 42
    assert metadata["datos"]["entrenamiento"] > 0
    assert metadata["datos"]["prueba"] > 0
    assert metadata["commit_codigo"]
    assert metadata["variables_entrada"]


def test_prediccion_consistente_y_una_por_fila(trained_version, tmp_path):
    model_path = trained_version["version_dir"] / "modelo.joblib"
    input_path = tmp_path / "entrada.csv"
    dataframe = make_dataset(rows=6, seed=3).drop(columns=["Survived"])
    dataframe.loc[0, "Age"] = np.nan
    dataframe.loc[1, "Embarked"] = np.nan
    dataframe.to_csv(input_path, index=False)

    result = run_prediction(model_path, input_path, tmp_path / "salida.csv")

    assert result["filas"] == len(dataframe)
    assert len(result["predicciones"]) == len(dataframe)
    assert "prediccion" in result["predicciones"].columns
    assert (tmp_path / "salida.csv").is_file()

    from src.artifacts import load_model

    model = load_model(model_path)
    expected = model.predict(coerce_feature_types(dataframe))
    assert np.array_equal(result["predicciones"]["prediccion"].to_numpy(), expected)
    assert result["predicciones"]["PassengerId"].tolist() == dataframe["PassengerId"].tolist()


def test_valores_faltantes_no_eliminan_filas(trained_version, tmp_path):
    model_path = trained_version["version_dir"] / "modelo.joblib"
    input_path = tmp_path / "nulos.csv"
    row = {
        "PassengerId": 1,
        "Pclass": 3,
        "Name": "Pasajero con nulos",
        "Sex": "male",
        "Age": None,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "T1",
        "Fare": None,
        "Cabin": None,
        "Embarked": None,
    }
    pd.DataFrame([row]).to_csv(input_path, index=False)

    result = run_prediction(model_path, input_path, tmp_path / "salida_nulos.csv")

    assert result["filas"] == 1
    assert len(result["predicciones"]) == 1
    assert result["predicciones"]["prediccion"].iloc[0] in (0, 1)


def test_columna_faltante_falla_con_error_claro(trained_version, tmp_path):
    model_path = trained_version["version_dir"] / "modelo.joblib"
    input_path = tmp_path / "invalido.csv"
    make_dataset(rows=4, seed=5).drop(columns=["Survived", "Age"]).to_csv(
        input_path, index=False
    )

    with pytest.raises(DataValidationError, match="Age"):
        run_prediction(model_path, input_path, tmp_path / "no_debe_existir.csv")

    completed = run_cli(
        [
            sys.executable,
            str(PREDICT_SCRIPT),
            "--model",
            str(model_path),
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "salida_fallida.csv"),
        ],
        cwd=PROJECT_ROOT,
    )
    assert completed.returncode != 0
    assert "Age" in completed.stderr
    assert not (tmp_path / "salida_fallida.csv").exists()


def test_versionamiento_no_sobrescribe(tmp_path, dataset_path):
    models_dir = tmp_path / "models"
    first = train_version(
        data_path=dataset_path, output_dir=models_dir, version="v1", seed=42
    )
    second = train_version(
        data_path=dataset_path, output_dir=models_dir, version="v2", seed=7
    )

    assert (models_dir / "v1" / "modelo.joblib").is_file()
    assert (models_dir / "v2" / "modelo.joblib").is_file()
    assert first["version_dir"] != second["version_dir"]
    assert first["version"] == "v1"
    assert second["version"] == "v2"

    original = (models_dir / "v1" / "modelo.joblib").read_bytes()
    with pytest.raises(VersionAlreadyExistsError):
        train_version(
            data_path=dataset_path, output_dir=models_dir, version="v1", seed=99
        )
    assert (models_dir / "v1" / "modelo.joblib").read_bytes() == original


def test_columnas_extra_ignoradas_y_orden_irrelevante(trained_version, tmp_path):
    model_path = trained_version["version_dir"] / "modelo.joblib"
    dataframe = make_dataset(rows=5, seed=11).drop(columns=["Survived"])
    dataframe["ColumnaExtra"] = "ignorar"
    shuffled = dataframe[list(reversed(dataframe.columns))]

    ordered_path = tmp_path / "ordenado.csv"
    shuffled_path = tmp_path / "desordenado.csv"
    dataframe.to_csv(ordered_path, index=False)
    shuffled.to_csv(shuffled_path, index=False)

    ordered = run_prediction(model_path, ordered_path, tmp_path / "a.csv")
    shuffled_result = run_prediction(model_path, shuffled_path, tmp_path / "b.csv")

    assert "ColumnaExtra" in ordered["columnas_ignoradas"]
    assert np.array_equal(
        ordered["predicciones"]["prediccion"].to_numpy(),
        shuffled_result["predicciones"]["prediccion"].to_numpy(),
    )


def test_cli_entrenamiento_genera_version(tmp_path, dataset_path):
    models_dir = tmp_path / "models"
    completed = run_cli(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--data",
            str(dataset_path),
            "--target",
            "Survived",
            "--output-dir",
            str(models_dir),
            "--version",
            "cli",
            "--seed",
            "42",
        ],
        cwd=PROJECT_ROOT,
    )

    assert completed.returncode == 0, completed.stderr
    assert (models_dir / "cli" / "modelo.joblib").is_file()
    assert (models_dir / "cli" / "metrics.json").is_file()
    assert (models_dir / "cli" / "metadata.json").is_file()

    duplicate = run_cli(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--data",
            str(dataset_path),
            "--output-dir",
            str(models_dir),
            "--version",
            "cli",
        ],
        cwd=PROJECT_ROOT,
    )
    assert duplicate.returncode != 0
    assert "ya existe" in duplicate.stderr
