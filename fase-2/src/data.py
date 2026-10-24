"""Lectura, validación y partición de los datos del Titanic."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    train_test_split,
)

from .config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from .errors import DataValidationError


def load_csv(path) -> pd.DataFrame:
    """Carga un CSV y falla con un mensaje claro si no existe o no se puede leer."""
    file_path = Path(path)
    if not file_path.is_file():
        raise DataValidationError(f"No se encontró el archivo de datos: {file_path}")
    try:
        return pd.read_csv(file_path)
    except Exception as exc:
        raise DataValidationError(
            f"No se pudo leer el archivo {file_path}: {exc}"
        ) from exc


def validate_feature_schema(df: pd.DataFrame, feature_columns=None) -> list:
    """Comprueba que existan todas las columnas de entrada requeridas."""
    features = list(feature_columns or FEATURE_COLUMNS)
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise DataValidationError(
            "Faltan columnas obligatorias: " + ", ".join(missing)
        )
    return features


def validate_target(df: pd.DataFrame, target: str) -> None:
    """Comprueba que la variable objetivo exista y sea binaria sin nulos."""
    if target not in df.columns:
        raise DataValidationError(
            f"No se encontró la variable objetivo '{target}' en el archivo de datos."
        )
    target_values = df[target]
    nulls = int(target_values.isna().sum())
    if nulls:
        raise DataValidationError(
            f"La variable objetivo '{target}' contiene {nulls} valores nulos."
        )
    if target_values.nunique() != 2:
        raise DataValidationError(
            f"La variable objetivo '{target}' debe tener exactamente dos clases."
        )


def coerce_feature_types(df: pd.DataFrame, feature_columns=None) -> pd.DataFrame:
    """Devuelve solo las variables de entrada, en orden canónico y con tipos válidos.

    Las columnas adicionales se ignoran. Una celda nula es válida: el pipeline
    guardado define cómo imputarla.
    """
    features = validate_feature_schema(df, feature_columns)
    ordered = df.loc[:, features].copy()

    for column in NUMERIC_FEATURES:
        if column not in ordered.columns:
            continue
        converted = pd.to_numeric(ordered[column], errors="coerce")
        invalid = converted.isna() & ordered[column].notna()
        if invalid.any():
            examples = ordered.loc[invalid, column].astype(str).unique()[:3]
            raise DataValidationError(
                f"La columna numérica '{column}' contiene valores no numéricos: "
                + ", ".join(examples)
            )
        ordered[column] = converted.astype(float)

    for column in CATEGORICAL_FEATURES:
        if column in ordered.columns:
            ordered[column] = ordered[column].astype("object")

    return ordered


def prepare_training_data(df: pd.DataFrame, target: str, feature_columns=None):
    """Separa la matriz de entrada y la variable objetivo para entrenar."""
    validate_target(df, target)
    features = validate_feature_schema(df, feature_columns)
    features_df = coerce_feature_types(df, features)
    target_series = df[target].astype(int)
    return features_df, target_series


def build_groups(dataframe: pd.DataFrame, group_column: str | None = None):
    """Agrupa pasajeros que comparten ticket para evitar fuga entre particiones.

    Los pasajeros sin `Ticket` reciben un grupo propio (no se agrupan entre sí
    por coincidencia de nulos). Devuelve None si la columna no está disponible.
    """
    if not group_column or group_column not in dataframe.columns:
        return None

    groups = dataframe[group_column].astype("string").str.strip()
    missing = groups.isna() | (groups == "")
    if missing.any():
        if "PassengerId" in dataframe.columns:
            fallback = "pasajero-" + dataframe["PassengerId"].astype("string")
        else:
            fallback = pd.Series(
                [f"fila-{position}" for position in range(len(dataframe))],
                index=dataframe.index,
                dtype="string",
            )
        groups = groups.mask(missing, fallback)

    return groups.astype("string")


def count_shared_groups(train_groups, test_groups) -> int:
    """Cuenta los grupos (tickets) presentes en entrenamiento y en prueba."""
    if train_groups is None or test_groups is None:
        return 0
    return len(set(train_groups.tolist()) & set(test_groups.tolist()))


def split_dataset(
    features: pd.DataFrame,
    target,
    test_size: float,
    seed: int,
    groups=None,
):
    """Divide entrenamiento/prueba sin fuga y sin partir grupos de viaje.

    Con grupos usa StratifiedGroupKFold (mantiene la proporción de clases y
    garantiza que un ticket no aparezca en ambos conjuntos). Sin grupos usa la
    partición estratificada clásica.
    """
    if groups is not None:
        folds = max(2, int(round(1 / test_size)))
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=seed
        )
        try:
            train_index, test_index = next(
                splitter.split(features, target, groups)
            )
        except ValueError as exc:
            raise DataValidationError(
                "No fue posible separar los datos por grupos: "
                f"{exc}. Revisa los grupos o cambia la semilla."
            ) from exc
        return (
            features.iloc[train_index],
            features.iloc[test_index],
            target.iloc[train_index],
            target.iloc[test_index],
        )

    try:
        return train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=seed,
            stratify=target,
        )
    except ValueError as exc:
        raise DataValidationError(
            f"No fue posible separar los datos en entrenamiento y prueba: {exc}"
        ) from exc
