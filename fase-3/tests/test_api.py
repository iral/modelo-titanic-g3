from fastapi.testclient import TestClient

from api import REPOSITORY_ROOT, create_app


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
