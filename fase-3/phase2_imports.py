"""Puente hacia las funciones compartidas de la Fase 2."""

import sys
from pathlib import Path

PHASE2_DIR = (Path(__file__).resolve().parent.parent / "fase-2").resolve()
if not PHASE2_DIR.is_dir():
    raise FileNotFoundError(f"No se encontró la carpeta de la Fase 2: {PHASE2_DIR}")

phase2_path = str(PHASE2_DIR)
if phase2_path not in sys.path:
    sys.path.insert(0, phase2_path)

from predict import predict_dataframe, run_prediction  # noqa: E402
from src.data import (  # noqa: E402
    coerce_feature_types,
    validate_feature_schema,
)
from train import train_version  # noqa: E402
