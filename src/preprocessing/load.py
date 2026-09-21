"""Carga del dataset crudo AI4I 2020."""

from pathlib import Path

import pandas as pd

COLUMNAS_REQUERIDAS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

N_REGISTROS_ESPERADOS = 10_000


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Carga el CSV crudo de AI4I 2020 y valida su integridad.

    Args:
        filepath: Ruta al CSV (normalmente `data/raw/ai4i2020.csv`).

    Returns:
        DataFrame con 10.000 filas y las 14 columnas originales del dataset.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan columnas, el número de filas no es el esperado
            o hay valores faltantes (FR-001, SC-001).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset no encontrado en {filepath}")

    df = pd.read_csv(filepath)

    faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas ausentes en el dataset: {faltantes}")

    if len(df) != N_REGISTROS_ESPERADOS:
        raise ValueError(
            f"Se esperaban {N_REGISTROS_ESPERADOS} registros, se encontraron {len(df)}"
        )

    n_nulos = int(df.isna().sum().sum())
    if n_nulos:
        raise ValueError(
            f"El dataset contiene {n_nulos} valores faltantes; resolver antes de modelar"
        )

    return df[COLUMNAS_REQUERIDAS]
