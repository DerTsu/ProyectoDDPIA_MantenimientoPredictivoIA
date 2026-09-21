"""Métricas de evaluación sobre la clase falla.

La constitución (Principio II) prohíbe accuracy como criterio de éxito: con un
desbalance de ~3% un clasificador que nunca predice falla alcanzaría 97% de
accuracy siendo inútil. Solo se reportan F1, Recall y PR-AUC de la clase minoría.
"""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, recall_score

UMBRAL_POR_DEFECTO = 0.5


def compute_metrics(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    threshold: float = UMBRAL_POR_DEFECTO,
) -> dict[str, float]:
    """Calcula F1, Recall y PR-AUC para la clase falla.

    PR-AUC se obtiene con `average_precision_score`, que suma los incrementos
    reales de la curva precisión-recall. Se prefiere sobre `auc()` sobre la curva
    porque la interpolación lineal del trapecio sobrestima el área en problemas
    con clase minoritaria pequeña.

    Args:
        y_true: Etiquetas reales binarias (n_samples,).
        y_proba_pos: Probabilidad predicha de la clase falla (n_samples,).
        threshold: Umbral para binarizar las probabilidades en F1 y Recall.

    Returns:
        Diccionario con las claves `f1`, `recall` y `pr_auc`, todas en [0, 1].
    """
    y_pred = (y_proba_pos >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_proba_pos)),
    }
