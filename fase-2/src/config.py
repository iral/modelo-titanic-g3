"""Constantes compartidas por entrenamiento y predicción."""

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
PRIMARY_METRIC = "f1"

OUTPUT_PREDICTION_COLUMN = "prediccion"
OUTPUT_PROBABILITY_COLUMN = "probabilidad_1"
TARGET_LABELS = {0: "no_sobrevivio", 1: "sobrevivio"}
