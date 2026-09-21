"""Preparación de vectores de entrada para el clasificador TabPFN-v2.

Funciones puras de preprocesamiento y validación de rangos, testables
sin inicializar Streamlit.
"""

import numpy as np

from src.preprocessing.preprocess import COLUMNAS_DUMMY, COLUMNAS_NUMERICAS

# Única fuente de verdad para el orden de features: la misma que usa
# `src.preprocessing.preprocess.preprocess_features()`. No hardcodear este orden
# aquí; si cambia en `src/`, este módulo debe seguirlo automáticamente.
NOMBRE_FEATURES = COLUMNAS_DUMMY + COLUMNAS_NUMERICAS

# Rangos válidos extraídos del dataset AI4I 2020
RANGOS_VALIDOS = {
    "air_temp": (290.0, 320.0),
    "process_temp": (300.0, 330.0),
    "rpm": (1000.0, 3000.0),
    "torque": (1.0, 100.0),
    "tool_wear": (0.0, 300.0),
}


def preprocess_input(
    product_type: str,
    air_temp: float,
    process_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float,
) -> np.ndarray:
    """Convierte los 6 inputs del formulario en el vector de 8 features del modelo.

    Aplica one-hot encoding a `product_type` (L, M, H) y concatena con las
    5 variables numéricas, ordenadas según `NOMBRE_FEATURES` (derivado de
    `src.preprocessing.preprocess`, la misma fuente que usa el pipeline de
    entrenamiento) para que ambos preprocesamientos no puedan desalinearse.

    Args:
        product_type: Tipo de producto (`"L"`, `"M"` o `"H"`).
        air_temp: Temperatura del aire en Kelvin.
        process_temp: Temperatura del proceso en Kelvin.
        rpm: Velocidad de rotación en RPM.
        torque: Torque en Newton-metro.
        tool_wear: Desgaste de herramienta en minutos.

    Returns:
        Array numpy de forma `(1, 8)` con los features en el orden del modelo.
    """
    valores_por_columna = {
        "Type_H": 1.0 if product_type == "H" else 0.0,
        "Type_L": 1.0 if product_type == "L" else 0.0,
        "Type_M": 1.0 if product_type == "M" else 0.0,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }
    return np.array(
        [[valores_por_columna[columna] for columna in NOMBRE_FEATURES]],
        dtype=float,
    )
