"""Tests unitarios para la interfaz Streamlit de mantenimiento predictivo.

Ejercitan las funciones puras de app.utils.physics y app.utils.preprocessing
sin necesidad de lanzar el servidor Streamlit. La estructura AAA sigue las
reglas de AGENTS.md.
"""

import math

import numpy as np
import pandas as pd
import pytest

from app.utils.physics import (
    PASO_UMBRAL_FALLA,
    UMBRAL_FALLA_MAX,
    UMBRAL_FALLA_MIN,
    UMBRAL_PRECAUCION,
    calculate_delta_t,
    calculate_power,
    get_alert_level,
)
from app.utils.preprocessing import preprocess_input
from src.preprocessing.preprocess import COLUMNA_TARGET, preprocess_features

# ── preprocess_input ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("product_type", ["L", "M", "H"])
def test_preprocess_input_produce_los_mismos_valores_que_preprocess_features_de_src(
    product_type,
):
    # 1. ARRANGE (una fila cruda concreta, expresada en el esquema de src y en los
    # 6 argumentos posicionales que recibe preprocess_input de la app; se parametriza
    # sobre los 3 tipos porque un cruce entre Type_H/Type_L/Type_M solo se nota cuando
    # el tipo evaluado no vale 0 en las tres columnas dummy)
    air_temp, process_temp, rpm, torque, tool_wear = 298.1, 308.6, 1551, 42.8, 7
    fila_cruda = pd.DataFrame(
        [
            {
                "Type": product_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
                COLUMNA_TARGET: 0,
            }
        ]
    )

    # 2. ACT (procesar la misma fila por los dos caminos de código independientes)
    vector_src = (
        preprocess_features(fila_cruda).drop(columns=[COLUMNA_TARGET]).to_numpy(dtype=float)
    )
    vector_app = preprocess_input(product_type, air_temp, process_temp, rpm, torque, tool_wear)

    # 3. ASSERT (mismos 8 valores, en el mismo orden, para la misma fila de entrada)
    np.testing.assert_array_equal(vector_src, vector_app)


def test_preprocess_input_tipo_l():
    # 1. ARRANGE
    product_type = "L"

    # 2. ACT
    resultado = preprocess_input(product_type, 298.1, 308.6, 1551, 42.8, 0)

    # 3. ASSERT
    assert resultado.shape == (1, 8)
    assert resultado[0, 0] == 0.0  # Type_H
    assert resultado[0, 1] == 1.0  # Type_L
    assert resultado[0, 2] == 0.0  # Type_M


def test_preprocess_input_tipo_m():
    # 1. ARRANGE
    product_type = "M"

    # 2. ACT
    resultado = preprocess_input(product_type, 298.1, 308.6, 1551, 42.8, 0)

    # 3. ASSERT
    assert resultado[0, 0] == 0.0
    assert resultado[0, 1] == 0.0
    assert resultado[0, 2] == 1.0


def test_preprocess_input_tipo_h():
    # 1. ARRANGE
    product_type = "H"

    # 2. ACT
    resultado = preprocess_input(product_type, 298.1, 308.6, 1551, 42.8, 0)

    # 3. ASSERT
    assert resultado[0, 0] == 1.0
    assert resultado[0, 1] == 0.0
    assert resultado[0, 2] == 0.0


def test_preprocess_input_valores_numericos():
    # 1. ARRANGE
    air_temp, process_temp, rpm, torque, tool_wear = 299.5, 309.2, 1400, 50.0, 120.0

    # 2. ACT
    resultado = preprocess_input("M", air_temp, process_temp, rpm, torque, tool_wear)

    # 3. ASSERT
    assert resultado[0, 3] == air_temp
    assert resultado[0, 4] == process_temp
    assert resultado[0, 5] == rpm
    assert resultado[0, 6] == torque
    assert resultado[0, 7] == tool_wear


def test_preprocess_input_es_numpy():
    # 1. ARRANGE
    product_type = "L"

    # 2. ACT
    resultado = preprocess_input(product_type, 298.1, 308.6, 1551, 42.8, 0)

    # 3. ASSERT
    assert isinstance(resultado, np.ndarray)
    assert resultado.dtype == float


def test_preprocess_input_one_hot_es_mutuamente_exclusivo():
    # 1. ARRANGE
    tipos = ["L", "M", "H"]

    # 2. ACT
    vectores = [preprocess_input(t, 298, 308, 1500, 40, 0)[0, :3] for t in tipos]

    # 3. ASSERT (cada vector tiene exactamente un 1 entre las primeras 3 posiciones)
    for v in vectores:
        assert v.sum() == 1.0


# ── get_alert_level ──────────────────────────────────────────────────────────


def test_alert_level_normal():
    # 1. ARRANGE
    probabilidad = 0.10

    # 2. ACT
    nivel = get_alert_level(probabilidad)

    # 3. ASSERT
    assert nivel == "Normal"


def test_alert_level_precaucion():
    # 1. ARRANGE
    probabilidad = 0.50

    # 2. ACT
    nivel = get_alert_level(probabilidad)

    # 3. ASSERT
    assert nivel == "Precaucion"


def test_alert_level_falla_inminente():
    # 1. ARRANGE
    probabilidad = 0.85

    # 2. ACT
    nivel = get_alert_level(probabilidad)

    # 3. ASSERT
    assert nivel == "Falla inminente"


def test_alert_level_limite_inferior():
    # 1. ARRANGE
    probabilidad = 0.0

    # 2. ACT
    nivel = get_alert_level(probabilidad)

    # 3. ASSERT
    assert nivel == "Normal"


def test_alert_level_limite_superior():
    # 1. ARRANGE
    probabilidad = 1.0

    # 2. ACT
    nivel = get_alert_level(probabilidad)

    # 3. ASSERT
    assert nivel == "Falla inminente"


def test_alert_level_en_umbral_precaucion():
    # 1. ARRANGE
    probabilidad = 0.30

    # 2. ACT
    nivel = get_alert_level(probabilidad)

    # 3. ASSERT (0.30 no es < 0.30, cae en Precaucion)
    assert nivel == "Precaucion"


def test_alert_level_en_umbral_falla():
    # 1. ARRANGE
    probabilidad = 0.70

    # 2. ACT
    nivel = get_alert_level(probabilidad)

    # 3. ASSERT (0.70 no es < 0.70, cae en Falla inminente)
    assert nivel == "Falla inminente"


def test_alert_level_umbral_de_falla_configurable():
    # 1. ARRANGE (Probabilidad 0.68 y dos umbrales de falla distintos)
    probabilidad = 0.68

    # 2. ACT (Evaluar con umbral 0.50 y 0.70)
    con_umbral_bajo = get_alert_level(probabilidad, umbral_falla=0.50)
    con_umbral_alto = get_alert_level(probabilidad, umbral_falla=0.70)

    # 3. ASSERT (0.68 supera 0.50 pero queda en Precaución con 0.70)
    assert con_umbral_bajo == "Falla inminente"
    assert con_umbral_alto == "Precaucion"


def test_alert_level_umbral_en_el_limite_de_falla():
    # 1. ARRANGE (probabilidad igual al umbral configurable)
    probabilidad = 0.55

    # 2. ACT (Evaluar con umbral de falla 0.55)
    nivel = get_alert_level(probabilidad, umbral_falla=0.55)

    # 3. ASSERT (>= umbral cae en Falla inminente)
    assert nivel == "Falla inminente"


def test_alert_level_umbral_bajo_mantiene_los_tres_niveles():
    # 1. ARRANGE (Una probabilidad de cada banda, con umbral de falla 0.40)
    casos = {"Normal": 0.20, "Precaucion": 0.35, "Falla inminente": 0.50}

    # 2. ACT (Evaluar cada caso con el umbral configurado)
    resultados = {
        esperado: get_alert_level(prob, umbral_falla=0.40) for esperado, prob in casos.items()
    }

    # 3. ASSERT (Las tres bandas siguen existiendo con el umbral configurable)
    assert all(resultados[esperado] == esperado for esperado in casos)


_PASO_PCT = round(PASO_UMBRAL_FALLA * 100)
_VALORES_SLIDER_PCT = range(
    round(UMBRAL_FALLA_MIN * 100), round(UMBRAL_FALLA_MAX * 100) + 1, _PASO_PCT
)


@pytest.mark.parametrize("umbral_falla_pct", _VALORES_SLIDER_PCT)
def test_alert_level_mantiene_los_tres_niveles_en_todo_el_rango_del_slider(umbral_falla_pct):
    # 1. ARRANGE (un umbral de falla válido del slider —en % entero, igual que
    # results.py lo construye— y una probabilidad representativa de cada banda)
    umbral_falla = umbral_falla_pct / 100
    probabilidad_normal = UMBRAL_PRECAUCION / 2
    probabilidad_precaucion = (UMBRAL_PRECAUCION + umbral_falla) / 2
    probabilidad_falla = umbral_falla

    # 2. ACT (evaluar el nivel de alerta de cada probabilidad con este umbral)
    nivel_normal = get_alert_level(probabilidad_normal, umbral_falla=umbral_falla)
    nivel_precaucion = get_alert_level(probabilidad_precaucion, umbral_falla=umbral_falla)
    nivel_falla = get_alert_level(probabilidad_falla, umbral_falla=umbral_falla)

    # 3. ASSERT (los tres niveles siguen siendo alcanzables, ninguno colapsa)
    assert nivel_normal == "Normal"
    assert nivel_precaucion == "Precaucion"
    assert nivel_falla == "Falla inminente"


# ── calculate_power ──────────────────────────────────────────────────────────


def test_potencia_cero_rpm():
    # 1. ARRANGE
    torque = 50.0
    rpm = 0.0

    # 2. ACT
    potencia = calculate_power(torque, rpm)

    # 3. ASSERT
    assert potencia == 0.0


def test_potencia_cero_torque():
    # 1. ARRANGE
    torque = 0.0
    rpm = 1500.0

    # 2. ACT
    potencia = calculate_power(torque, rpm)

    # 3. ASSERT
    assert potencia == 0.0


def test_potencia_valores_conocidos():
    # 1. ARRANGE (torque=10 Nm, rpm=60 → ω = 2π rad/s → P = 20π ≈ 62.83 W)
    torque = 10.0
    rpm = 60.0
    esperado = 10.0 * 2.0 * math.pi

    # 2. ACT
    potencia = calculate_power(torque, rpm)

    # 3. ASSERT
    assert math.isclose(potencia, esperado, rel_tol=1e-9)


def test_potencia_rpm_altos():
    # 1. ARRANGE
    torque = 42.8
    rpm = 1551.0

    # 2. ACT
    potencia = calculate_power(torque, rpm)

    # 3. ASSERT (P > 0 y en un rango razonable)
    assert potencia > 0
    assert potencia < 100_000  # límite superior de sanity check


def test_potencia_es_escalar():
    # 1. ARRANGE
    torque = 40.0
    rpm = 1500.0

    # 2. ACT
    potencia = calculate_power(torque, rpm)

    # 3. ASSERT
    assert isinstance(potencia, float)


# ── calculate_delta_t ────────────────────────────────────────────────────────


def test_delta_t_proceso_mayor():
    # 1. ARRANGE
    process_temp = 308.6
    air_temp = 298.1

    # 2. ACT
    delta = calculate_delta_t(process_temp, air_temp)

    # 3. ASSERT
    assert delta == pytest.approx(10.5)


def test_delta_t_iguales():
    # 1. ARRANGE
    temp = 300.0

    # 2. ACT
    delta = calculate_delta_t(temp, temp)

    # 3. ASSERT
    assert delta == 0.0


def test_delta_t_aire_mayor():
    # 1. ARRANGE
    process_temp = 295.0
    air_temp = 305.0

    # 2. ACT
    delta = calculate_delta_t(process_temp, air_temp)

    # 3. ASSERT
    assert delta == -10.0
