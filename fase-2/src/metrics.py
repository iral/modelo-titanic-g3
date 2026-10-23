"""Cálculo de métricas de clasificación para el modelo y el baseline."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Calcula las métricas de la Fase 1 para la clase positiva (1 = sobrevivió)."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = None

    return metrics


def evaluate_estimator(estimator, features, target) -> dict:
    """Evalúa un estimador y devuelve métricas listas para serializar a JSON."""
    predictions = np.asarray(estimator.predict(features))
    probabilities = None
    if hasattr(estimator, "predict_proba"):
        try:
            probabilities = np.asarray(estimator.predict_proba(features))[:, 1]
        except (AttributeError, IndexError):
            probabilities = None
    return compute_classification_metrics(target, predictions, probabilities)
