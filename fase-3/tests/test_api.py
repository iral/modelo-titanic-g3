import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api as api_module
from api import REPOSITORY_ROOT, create_app

PREDICTION_INPUT = {
    "Pclass": 3,
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Sex": "male",
    "Embarked": "S",
}


def test_health_uses_default_model_version(monkeypatch):
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    monkeypatch.delenv("MODELS_DIR", raising=False)
    app = create_app()

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "v1"}
    assert app.state.model_version_dir == REPOSITORY_ROOT / "fase-2" / "models" / "v1"


def test_health_uses_configured_model_version(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v2")
    app = create_app()

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "v2"}
    assert app.state.model_version_dir == REPOSITORY_ROOT / "fase-2" / "models" / "v2"


def test_models_dir_resolves_relative_path_from_repository_root(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v2")
    monkeypatch.setenv("MODELS_DIR", "custom-models")

    app = create_app()

    assert app.state.models_dir == REPOSITORY_ROOT / "custom-models"
    assert app.state.model_version_dir == REPOSITORY_ROOT / "custom-models" / "v2"


def test_models_dir_preserves_absolute_path(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_VERSION", "v1")
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))

    app = create_app()

    assert app.state.models_dir == tmp_path.resolve()
    assert app.state.model_version_dir == tmp_path.resolve() / "v1"


class RecordingModel:
    def __init__(self, prediction=1):
        self.prediction = prediction
        self.features = None

    def predict(self, features):
        self.features = features.copy()
        return np.array([self.prediction])


def test_predict_returns_binary_prediction_and_configured_version(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MODEL_VERSION", "test-version")
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    model = RecordingModel(prediction=1)
    loaded_paths = []

    def load_model(model_path):
        loaded_paths.append(model_path)
        return model

    monkeypatch.setattr(api_module, "load_model", load_model)
    app = create_app()

    response = TestClient(app).post("/predict", json=PREDICTION_INPUT)

    assert response.status_code == 200
    assert set(response.json()) == {"prediccion", "version"}
    assert response.json() == {"prediccion": 1, "version": "test-version"}
    assert isinstance(response.json()["prediccion"], int)
    assert response.json()["prediccion"] in (0, 1)
    assert loaded_paths == [app.state.model_version_dir]
    assert list(model.features.columns) == [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Sex",
        "Embarked",
    ]


def test_predict_preserves_null_values_for_pipeline(monkeypatch):
    model = RecordingModel(prediction=0)
    monkeypatch.setattr(api_module, "load_model", lambda _model_path: model)
    payload = {
        **PREDICTION_INPUT,
        "Age": None,
        "Fare": None,
        "Embarked": None,
    }

    response = TestClient(create_app()).post("/predict", json=payload)

    assert response.status_code == 200
    assert pd.isna(model.features.loc[0, "Age"])
    assert pd.isna(model.features.loc[0, "Fare"])
    assert pd.isna(model.features.loc[0, "Embarked"])


def test_predict_missing_required_field_returns_422_with_field_name():
    payload = {key: value for key, value in PREDICTION_INPUT.items() if key != "Age"}

    response = TestClient(create_app()).post("/predict", json=payload)

    assert response.status_code == 422
    assert "Age" in response.text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Age", "22"),
        ("Pclass", True),
        ("Sex", 1),
    ],
)
def test_predict_incompatible_field_type_returns_422_with_field_name(field, value):
    payload = {**PREDICTION_INPUT, field: value}

    response = TestClient(create_app()).post("/predict", json=payload)

    assert response.status_code == 422
    assert field in response.text


def test_predict_rejects_additional_field():
    payload = {**PREDICTION_INPUT, "Survived": 1}

    response = TestClient(create_app()).post("/predict", json=payload)

    assert response.status_code == 422
    assert "Survived" in response.text


def test_predict_missing_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_VERSION", "missing")
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))

    response = TestClient(create_app()).post("/predict", json=PREDICTION_INPUT)

    assert response.status_code == 503
    assert "modelo" in response.json()["detail"].lower()


def test_predict_with_saved_model_pipeline(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v1")
    monkeypatch.setenv("MODELS_DIR", str(REPOSITORY_ROOT / "fase-2" / "models"))
    payload = {**PREDICTION_INPUT, "Age": None, "Embarked": None}

    response = TestClient(create_app()).post("/predict", json=payload)

    assert response.status_code == 200
    assert set(response.json()) == {"prediccion", "version"}
    assert response.json()["version"] == "v1"
    assert response.json()["prediccion"] in (0, 1)
