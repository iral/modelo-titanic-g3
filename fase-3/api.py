"""API de Fase 3."""

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, StrictStr

from phase2_imports import coerce_feature_types, validate_feature_schema
from src.artifacts import load_model
from src.config import FEATURE_COLUMNS
from src.errors import ModelLoadError, ModelNotFoundError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TitanicPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Pclass: StrictInt | StrictFloat | None
    Age: StrictInt | StrictFloat | None
    SibSp: StrictInt | StrictFloat | None
    Parch: StrictInt | StrictFloat | None
    Fare: StrictInt | StrictFloat | None
    Sex: StrictStr | None
    Embarked: StrictStr | None


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

    @application.post("/predict")
    def predict(
        passenger: TitanicPrediction, request: Request
    ) -> dict[str, int | str]:
        dataframe = pd.DataFrame([passenger.model_dump()])
        dataframe = dataframe.where(dataframe.notna(), float("nan"))
        validate_feature_schema(dataframe, FEATURE_COLUMNS)
        features = coerce_feature_types(dataframe, FEATURE_COLUMNS)

        try:
            model = load_model(request.app.state.model_version_dir)
        except (ModelNotFoundError, ModelLoadError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"No se pudo cargar el modelo configurado: {exc}",
            ) from exc

        prediction = model.predict(features)[0]
        return {
            "prediccion": int(prediction),
            "version": request.app.state.model_version,
        }

    return application


app = create_app()
