"""Constantes compartidas por entrenamiento y predicción."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data" / "raw" / "train.csv"

TARGET_DEFAULT = "Survived"

NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DEFAULT_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
}

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5
DEFAULT_SELECT_BY = "f1"
DEFAULT_GROUP_COLUMN = "Ticket"
PRIMARY_METRIC = "f1"

OUTPUT_PREDICTION_COLUMN = "prediccion"
OUTPUT_PROBABILITY_COLUMN = "probabilidad_1"
TARGET_LABELS = {0: "no_sobrevivio", 1: "sobrevivio"}
