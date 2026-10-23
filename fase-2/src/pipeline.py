"""Construcción del preprocesamiento y del estimador.

El preprocesamiento y el clasificador viven en un único Pipeline para que las
reglas aprendidas durante el entrenamiento se guarden y reutilicen al predecir.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL_PARAMS,
    NUMERIC_FEATURES,
)


def build_preprocessor(
    numeric_features=None,
    categorical_features=None,
) -> ColumnTransformer:
    """Imputa y codifica las variables usando la estrategia de la Fase 1.

    - Numéricas: imputación por mediana.
    - Categóricas: imputación por categoría más frecuente y OneHotEncoder con
      handle_unknown="ignore" para tolerar categorías nuevas al predecir.
    """
    numeric = list(numeric_features or NUMERIC_FEATURES)
    categorical = list(categorical_features or CATEGORICAL_FEATURES)

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ]
    )


def build_estimator(seed: int = 42, params: dict | None = None) -> RandomForestClassifier:
    """Crea el RandomForestClassifier con los hiperparámetros indicados."""
    merged = {**DEFAULT_MODEL_PARAMS, **(params or {})}
    merged["random_state"] = seed
    return RandomForestClassifier(**merged)


def build_pipeline(
    seed: int = 42,
    params: dict | None = None,
    numeric_features=None,
    categorical_features=None,
) -> Pipeline:
    """Devuelve el pipeline completo (preprocesamiento + clasificador)."""
    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            ("classifier", build_estimator(seed, params)),
        ]
    )
