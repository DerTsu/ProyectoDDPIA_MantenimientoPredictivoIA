"""Preprocesamiento mínimo del dataset AI4I 2020 (Principio III de la constitución)."""

import pandas as pd

COLUMNA_TARGET = "Machine failure"

COLUMNAS_NUMERICAS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

COLUMNAS_DUMMY = ["Type_H", "Type_L", "Type_M"]

# Identificadores y submodos de falla: fuera de alcance para el target binario (spec 001).
COLUMNAS_DESCARTADAS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica one-hot a `Type` y descarta columnas fuera de alcance.

    No muta el DataFrame de entrada.

    Args:
        df: DataFrame crudo devuelto por `load_raw_data()`.

    Returns:
        DataFrame de 9 columnas: Type_H, Type_L, Type_M, las 5 variables de
        proceso y el target `Machine failure`.
    """
    dummies = pd.get_dummies(df["Type"], prefix="Type", dtype=float)
    dummies = dummies.reindex(columns=COLUMNAS_DUMMY, fill_value=0.0)

    procesado = pd.concat([dummies, df[COLUMNAS_NUMERICAS], df[[COLUMNA_TARGET]]], axis=1)
    return procesado[COLUMNAS_DUMMY + COLUMNAS_NUMERICAS + [COLUMNA_TARGET]]
