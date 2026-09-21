"""Construcción de matrices de entrenamiento y prueba con split estratificado."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.preprocess import COLUMNA_TARGET, preprocess_features

TEST_SIZE = 0.2
RANDOM_STATE = 42


def build_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocesa y divide el dataset en train/test estratificado 80-20.

    La estratificación preserva la proporción de la clase falla (~3%) en ambos
    conjuntos, requisito para que las métricas de la clase minoría sean estables
    (FR-003, SC-003).

    Args:
        df: DataFrame crudo devuelto por `load_raw_data()`.

    Returns:
        Tupla (X_train, y_train, X_test, y_test) con formas
        (8000, 8), (8000,), (2000, 8), (2000,).
    """
    procesado = preprocess_features(df)

    X = procesado.drop(columns=[COLUMNA_TARGET]).to_numpy(dtype=float)
    y = procesado[COLUMNA_TARGET].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, y_train, X_test, y_test
