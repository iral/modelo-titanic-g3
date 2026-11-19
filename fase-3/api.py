"""API de Fase 3."""

import os
from pathlib import Path

from fastapi import FastAPI

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def create_app() -> FastAPI:
    model_version = os.getenv("MODEL_VERSION", "v1")
    models_dir = Path(os.getenv("MODELS_DIR", "fase-2/models"))
    if not models_dir.is_absolute():
        models_dir = REPOSITORY_ROOT / models_dir
    models_dir = models_dir.resolve()
    model_version_dir = models_dir / model_version

    application = FastAPI()
    application.state.model_version = model_version
    application.state.models_dir = models_dir
    application.state.model_version_dir = model_version_dir

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": model_version}

    return application


app = create_app()
