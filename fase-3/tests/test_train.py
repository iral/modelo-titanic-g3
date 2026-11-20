"""Pruebas de POST /train y de su integración con POST /predict.

Qué es real y qué es simulado:
- REAL: el entrenamiento de la Fase 2 (``train_version``), la escritura de
  artefactos, la carga del modelo y la predicción.
- Reducido por velocidad: las pruebas fijan ``model_params`` pequeños
  (10 árboles) en lugar de ejecutar la búsqueda completa de hiperparámetros.
- Simulado: solo los fallos (excepción inyectada) en las pruebas de error.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import training_service
from api import create_app
from src.artifacts import load_model

FAST_PARAMS = {"n_estimators": 10, "max_depth": 3}
FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
PASSENGER = {
    "Pclass": 3, "Age": 22.0, "SibSp": 1, "Parch": 0,
    "Fare": 7.25, "Sex": "male", "Embarked": "S",
}


def make_dataset(rows: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dataframe = pd.DataFrame(
        {
            "PassengerId": np.arange(1, rows + 1),
            "Pclass": rng.integers(1, 4, rows),
            "Name": [f"Pasajero {i}" for i in range(rows)],
            "Sex": rng.choice(["male", "female"], rows),
            "Age": np.round(rng.normal(30, 10, rows), 1),
            "SibSp": rng.integers(0, 3, rows),
            "Parch": rng.integers(0, 3, rows),
            "Ticket": [f"T{i // 3}" for i in range(rows)],
            "Fare": np.round(rng.uniform(5, 200, rows), 2),
            "Cabin": rng.choice([None, "C1", "B2"], rows),
            "Embarked": rng.choice(["S", "C", "Q"], rows),
            "Survived": rng.integers(0, 2, rows),
        }
    )
    dataframe.loc[0:4, "Age"] = np.nan
    return dataframe


@pytest.fixture()
def env(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "train.csv"
    make_dataset().to_csv(csv_path, index=False)

    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("MODEL_VERSION", "v1")

    real_train = training_service.train_version  # entrenamiento real de la Fase 2
    monkeypatch.setattr(
        training_service,
        "train_version",
        lambda **kwargs: real_train(model_params=FAST_PARAMS, **kwargs),
    )
    return SimpleNamespace(
        models=models_dir, data=data_dir, csv=csv_path, real_train=real_train
    )


def post_train(env, **body):
    return TestClient(create_app()).post(
        "/train", json={"data_path": str(env.csv), **body}
    )


def versions(models_dir):
    if not models_dir.exists():
        return []
    return sorted(p.name for p in models_dir.iterdir())


# ---------- Issue 1: entrenamiento y versionado ----------

def test_train_valido_crea_version_con_artefactos(env):
    response = post_train(env, seed=42)

    assert response.status_code == 201
    body = response.json()
    assert body["version"] == "v1"
    assert body["semilla"] == 42
    assert "f1" in body["metricas"]["modelo"]
    for name in ("modelo.joblib", "metrics.json", "metadata.json"):
        assert (env.models / "v1" / name).is_file()
    assert versions(env.models) == ["v1"]  # sin restos temporales


def test_train_sin_cuerpo_usa_configuracion_por_defecto(env, monkeypatch):
    monkeypatch.delenv("DATA_DIR")
    response = TestClient(create_app()).post("/train")  # dataset real del repositorio

    assert response.status_code == 201
    assert response.json()["variable_objetivo"] == "Survived"
    assert response.json()["version"] == "v1"


def test_versiones_son_unicas_y_no_sobrescriben_anteriores(env):
    (env.models / "v1").mkdir(parents=True)
    (env.models / "v1" / "modelo.joblib").write_bytes(b"original")

    first = post_train(env)
    second = post_train(env)

    assert first.json()["version"] == "v2"
    assert second.json()["version"] == "v3"
    assert (env.models / "v1" / "modelo.joblib").read_bytes() == b"original"
    assert versions(env.models) == ["v1", "v2", "v3"]


def test_train_no_cambia_model_version(env):
    client = TestClient(create_app())
    client.post("/train", json={"data_path": str(env.csv)})

    assert client.get("/health").json() == {"status": "ok", "version": "v1"}


# ---------- Issue 2: errores ----------

@pytest.mark.parametrize(
    ("body", "campo"),
    [
        ({"seed": "abc"}, "seed"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"target": ""}, "target"),
        ({"model_params": {"n_estimators": 5000}}, "model_params"),
    ],
)
def test_configuracion_invalida_devuelve_422_con_campo(env, body, campo):
    response = TestClient(create_app()).post("/train", json=body)

    assert response.status_code == 422
    assert campo in response.text
    assert versions(env.models) == []


def test_archivo_inexistente_devuelve_404(env):
    response = TestClient(create_app()).post(
        "/train", json={"data_path": str(env.data / "no_existe.csv")}
    )

    assert response.status_code == 404
    assert "no_existe.csv" in response.json()["detail"]
    assert versions(env.models) == []


def test_ruta_fuera_del_directorio_de_datos_devuelve_422(env, tmp_path):
    outside = tmp_path / "otro.csv"
    make_dataset().to_csv(outside, index=False)

    response = TestClient(create_app()).post("/train", json={"data_path": str(outside)})

    assert response.status_code == 422
    assert "data_path" in response.json()["detail"]


def test_sin_variable_objetivo_devuelve_422(env):
    make_dataset().drop(columns=["Survived"]).to_csv(env.csv, index=False)

    response = post_train(env)

    assert response.status_code == 422
    assert "Survived" in response.json()["detail"]
    assert versions(env.models) == []


def test_columna_predictora_ausente_devuelve_422(env):
    make_dataset().drop(columns=["Age"]).to_csv(env.csv, index=False)

    response = post_train(env)

    assert response.status_code == 422
    assert "Age" in response.json()["detail"]


def test_fallo_de_entrenamiento_no_publica_version_y_api_sigue(env, monkeypatch):
    (env.models / "v1").mkdir(parents=True)
    (env.models / "v1" / "modelo.joblib").write_bytes(b"original")

    def boom(**_kwargs):
        raise RuntimeError("detalle interno secreto")

    monkeypatch.setattr(training_service, "train_version", boom)
    client = TestClient(create_app())
    response = client.post("/train", json={"data_path": str(env.csv)})

    assert response.status_code == 500
    assert "secreto" not in response.text
    assert versions(env.models) == ["v1"]
    assert (env.models / "v1" / "modelo.joblib").read_bytes() == b"original"
    assert client.get("/health").status_code == 200


def test_fallo_al_publicar_no_deja_version_parcial(env, monkeypatch):
    def boom(*_args):
        raise OSError("disco lleno")

    monkeypatch.setattr(training_service, "_publish_version", boom)
    response = post_train(env)

    assert response.status_code == 500
    assert "no pudo completarse" in response.json()["detail"]
    assert versions(env.models) == []  # tampoco quedan directorios temporales


# ---------- Issue 3: integración /train -> /predict ----------

def test_ciclo_completo_entrenar_seleccionar_y_predecir(env, monkeypatch):
    assert post_train(env, seed=1).json()["version"] == "v1"
    client_v1 = TestClient(create_app())  # MODEL_VERSION=v1
    first = client_v1.post("/predict", json=PASSENGER)
    assert first.status_code == 200 and first.json()["version"] == "v1"

    assert post_train(env, seed=2).json()["version"] == "v2"
    assert (env.models / "v1" / "modelo.joblib").is_file()
    # sin reconfigurar, sigue usando v1
    assert client_v1.post("/predict", json=PASSENGER).json()["version"] == "v1"

    monkeypatch.setenv("MODEL_VERSION", "v2")
    second = TestClient(create_app()).post("/predict", json=PASSENGER)
    assert second.status_code == 200
    assert second.json()["version"] == "v2"
    assert second.json()["prediccion"] in (0, 1)


def test_cada_version_es_autocontenida_y_coherente(env):
    post_train(env, seed=1)
    post_train(env, seed=2)

    import json

    for version in ("v1", "v2"):
        metadata = json.loads((env.models / version / "metadata.json").read_text("utf-8"))
        assert metadata["version"] == version
        model = load_model(env.models / version)
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            assert list(names) == FEATURES
    seeds = [
        json.loads((env.models / v / "metadata.json").read_text("utf-8"))["semilla"]
        for v in ("v1", "v2")
    ]
    assert seeds == [1, 2]


def test_version_inexistente_devuelve_503_sin_usar_otra(env, monkeypatch):
    post_train(env)
    monkeypatch.setenv("MODEL_VERSION", "v9")

    response = TestClient(create_app()).post("/predict", json=PASSENGER)

    assert response.status_code == 503
    assert "modelo" in response.json()["detail"].lower()


@pytest.mark.parametrize("target", ["Pclass", "Sex", "Age", "Fare", "NoExiste"])
def test_target_distinto_de_survived_devuelve_422(env, target):
    response = post_train(env, target=target)
    assert response.status_code == 422
    assert "Survived" in response.json()["detail"]
    assert versions(env.models) == []


def test_target_con_valores_distintos_de_cero_y_uno_devuelve_422(env):
    dataframe = make_dataset()
    dataframe.loc[0, "Survived"] = 2
    dataframe.to_csv(env.csv, index=False)
    response = post_train(env)
    assert response.status_code == 422
    assert "0 y 1" in response.json()["detail"]
    assert versions(env.models) == []


def test_target_con_nulos_devuelve_422(env):
    dataframe = make_dataset()
    dataframe.loc[0, "Survived"] = np.nan
    dataframe.to_csv(env.csv, index=False)
    response = post_train(env)
    assert response.status_code == 422
    assert "nulos" in response.json()["detail"]
    assert versions(env.models) == []


def test_data_dir_personalizado_usa_raw_train_csv(env, monkeypatch):
    custom_dir = env.data / "custom-data"
    raw_dir = custom_dir / "raw"
    raw_dir.mkdir(parents=True)
    make_dataset().to_csv(raw_dir / "train.csv", index=False)
    monkeypatch.setenv("DATA_DIR", str(custom_dir))
    response = TestClient(create_app()).post("/train")
    assert response.status_code == 201
    assert response.json()["version"] == "v1"


def test_data_path_no_puede_escapar_mediante_enlace_simbolico(env, tmp_path):
    outside = tmp_path / "fuera.csv"
    make_dataset().to_csv(outside, index=False)
    link = env.data / "enlace.csv"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("El sistema no permite crear enlaces simbólicos.")

    response = TestClient(create_app()).post("/train", json={"data_path": str(link)})
    assert response.status_code == 422
    assert "data_path" in response.json()["detail"]


@pytest.mark.parametrize("failure", [ValueError("dato inválido"), TypeError("tipo inválido")])
def test_errores_de_datos_de_fase2_se_mapean_a_422(env, monkeypatch, failure):
    def fail(**_kwargs):
        raise failure
    monkeypatch.setattr(training_service, "train_version", fail)
    response = TestClient(create_app()).post(
        "/train", json={"data_path": str(env.csv)}
    )
    assert response.status_code == 422
    assert "datos" in response.json()["detail"].lower()
    assert versions(env.models) == []
