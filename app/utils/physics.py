"""Funciones puras de ingeniería física para mantenimiento predictivo.

Fórmulas de ΔT y potencia mecánica, testables sin inicializar Streamlit.
"""

import numpy as np

UMBRAL_PRECAUCION = 0.30
UMBRAL_FALLA = 0.70
UMBRAL_FALLA_MAX = 0.90
PASO_UMBRAL_FALLA = 0.05
UMBRAL_FALLA_MIN = UMBRAL_PRECAUCION + PASO_UMBRAL_FALLA


def calculate_power(torque: float, rpm: float) -> float:
    """Calcula la potencia mecánica en watts.

    Fórmula: P = τ · ω, donde ω = 2π · rpm / 60.

    Args:
        torque: Torque en Newton-metro.
        rpm: Velocidad de rotación en revoluciones por minuto.

    Returns:
        Potencia en watts.
    """
    omega = 2.0 * np.pi * rpm / 60.0
    return torque * omega


def calculate_delta_t(process_temp: float, air_temp: float) -> float:
    """Calcula la diferencia de temperatura entre proceso y aire.

    Args:
        process_temp: Temperatura del proceso en Kelvin.
        air_temp: Temperatura del aire en Kelvin.

    Returns:
        ΔT en Kelvin (proceso − aire).
    """
    return process_temp - air_temp


def get_alert_level(probability: float, umbral_falla: float = UMBRAL_FALLA) -> str:
    """Determina el nivel de alerta según la probabilidad de falla.

    Mantiene la banda de Precaución fija por debajo de `UMBRAL_PRECAUCION`;
    el umbral de Falla inminente es configurable. Con el valor por defecto
    (0.70) el comportamiento es idéntico al esquema original de tres niveles.

    Args:
        probability: Probabilidad predicha de falla en `[0, 1]`.
        umbral_falla: Umbral de decisión de Falla inminente (mayor o igual a
            `UMBRAL_PRECAUCION` para conservar las tres bandas).

    Returns:
        `"Normal"`, `"Precaucion"` o `"Falla inminente"`.
    """
    if probability < UMBRAL_PRECAUCION:
        return "Normal"
    if probability < umbral_falla:
        return "Precaucion"
    return "Falla inminente"
