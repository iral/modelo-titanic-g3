"""API de Fase 3 para predicción y entrenamiento versionado del Titanic."""

import os
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from phase2_imports import predict_dataframe
from src.artifacts import load_model
from src.config import FEATURE_COLUMNS
from src.errors import DataValidationError, ModelLoadError, ModelNotFoundError
from training_service import (
    ArtifactSaveError,
    DataFileNotFoundError,
    InvalidTrainingConfigError,
    InvalidTrainingDataError,
    TrainingFailedError,
    train_new_version,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TitanicPrediction(BaseModel):
    """Esquema estricto de entrada; los nulos se delegan al imputador del pipeline."""

    model_config = ConfigDict(extra="forbid")

    Pclass: StrictInt | None = Field(ge=1, le=3)
    Age: StrictInt | StrictFloat | None = Field(ge=0)
    SibSp: StrictInt | None = Field(ge=0)
    Parch: StrictInt | None = Field(ge=0)
    Fare: StrictInt | StrictFloat | None = Field(ge=0)
    Sex: Literal["male", "female"] | None
    Embarked: Literal["S", "C", "Q"] | None


class TrainRequest(BaseModel):
    """Cuerpo opcional de POST /train. Sin cuerpo se usa la configuración del proyecto."""

    model_config = ConfigDict(extra="forbid")

    data_path: str | None = Field(default=None, min_length=1, max_length=500)
    target: str | None = Field(default=None, min_length=1, max_length=64)
    seed: StrictInt | None = Field(default=None, ge=0, le=2**32 - 1)


def create_app() -> FastAPI:
    model_version = os.getenv("MODEL_VERSION", "v1")
    models_dir = Path(os.getenv("MODELS_DIR", "fase-2/models"))
    if not models_dir.is_absolute():
        models_dir = REPOSITORY_ROOT / models_dir
    models_dir = models_dir.resolve()
    model_version_dir = models_dir / model_version

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = REPOSITORY_ROOT / data_dir
    data_dir = data_dir.resolve()

    application = FastAPI()
    application.state.model_version = model_version
    application.state.models_dir = models_dir
    application.state.model_version_dir = model_version_dir
    application.state.data_dir = data_dir

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": model_version}

    @application.post("/predict")
    def predict(passenger: TitanicPrediction, request: Request) -> dict[str, int | str]:
        dataframe = pd.DataFrame([passenger.model_dump()])
        dataframe = dataframe.where(dataframe.notna(), float("nan"))

        try:
            # Carga el modelo de la versión seleccionada; la inferencia y el
            # procesamiento de features son compartidos con fase-2/predict.py.
            model = load_model(request.app.state.model_version_dir)
        except (ModelNotFoundError, ModelLoadError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"No se pudo cargar el modelo configurado: {exc}",
            ) from exc

        try:
            result = predict_dataframe(model, dataframe)
        except DataValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        prediction = result["predicciones"].iloc[0]["prediccion"]
        return {
            "prediccion": int(prediction),
            "version": request.app.state.model_version,
        }

    @application.post("/train", status_code=201)
    def train(request: Request, body: TrainRequest | None = None) -> dict:
        body = body or TrainRequest()
        try:
            result = train_new_version(
                models_dir=request.app.state.models_dir,
                data_dir=request.app.state.data_dir,
                data_path=body.data_path,
                target=body.target,
                seed=body.seed,
            )
        except DataFileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (InvalidTrainingConfigError, InvalidTrainingDataError, DataValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ArtifactSaveError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "El entrenamiento no pudo completarse correctamente: "
                    f"{exc} No se publicó ninguna versión nueva."
                ),
            ) from exc
        except TrainingFailedError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"{exc} No se publicó ninguna versión nueva.",
            ) from exc

        return {
            "mensaje": "Entrenamiento completado",
            # Entrenar no activa automáticamente la nueva versión.
            "version_activa": request.app.state.model_version,
            **result,
        }

    return application


app = create_app()
