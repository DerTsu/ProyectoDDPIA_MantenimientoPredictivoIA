"""T020 [US3] Tests unitarios para el baseline XGBoost."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.models import XGBoostBaseline


@pytest.fixture(scope="module")
def datos_sinteticos():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 8))
    y_train = (rng.random(200) < 0.2).astype(int)
    X_test = rng.normal(size=(40, 8))
    return X_train, y_train, X_test


def test_se_instancia_sin_errores():
    # 1. ARRANGE (Peso de la clase minoritaria conocido)
    peso = 30.0

    # 2. ACT (Instanciar el baseline)
    modelo = XGBoostBaseline(scale_pos_weight=peso)

    # 3. ASSERT (Conserva el peso recibido)
    assert modelo.scale_pos_weight == peso


def test_from_class_balance_calcula_el_peso():
    # 1. ARRANGE (Etiquetas con 970 negativos y 30 positivos)
    y = np.array([0] * 970 + [1] * 30)

    # 2. ACT (Construir el modelo desde el balance de clases)
    modelo = XGBoostBaseline.from_class_balance(y)

    # 3. ASSERT (scale_pos_weight = negativos / positivos)
    assert modelo.scale_pos_weight == pytest.approx(970 / 30)


def test_from_class_balance_sin_positivos_lanza_error():
    # 1. ARRANGE (Etiquetas sin ningún caso de falla)
    y = np.zeros(100, dtype=int)

    # 2. ACT + 3. ASSERT (No se puede ponderar una clase ausente)
    with pytest.raises(ValueError, match="clase falla"):
        XGBoostBaseline.from_class_balance(y)


def test_fit_y_predict_proba(datos_sinteticos):
    # 1. ARRANGE (Datos sintéticos y modelo ponderado por el desbalance)
    X_train, y_train, X_test = datos_sinteticos
    modelo = XGBoostBaseline.from_class_balance(y_train)

    # 2. ACT (Entrenar y predecir probabilidades)
    modelo.fit(X_train, y_train)
    proba = modelo.predict_proba(X_test)

    # 3. ASSERT (Matriz (n, 2) de probabilidades válidas)
    assert proba.shape == (len(X_test), 2)
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predecir_sin_entrenar_falla(datos_sinteticos):
    # 1. ARRANGE (Modelo recién instanciado, sin entrenar)
    _, _, X_test = datos_sinteticos
    modelo = XGBoostBaseline(scale_pos_weight=4.0)

    # 2. ACT + 3. ASSERT (Predecir sin fit debe fallar explícitamente)
    with pytest.raises(NotFittedError):
        modelo.predict_proba(X_test)


def test_es_reproducible(datos_sinteticos):
    # 1. ARRANGE (Mismos datos y misma semilla en dos entrenamientos)
    X_train, y_train, X_test = datos_sinteticos

    # 2. ACT (Entrenar y predecir dos veces por separado)
    predicciones = []
    for _ in range(2):
        modelo = XGBoostBaseline(scale_pos_weight=4.0, random_state=42)
        modelo.fit(X_train, y_train)
        predicciones.append(modelo.predict_proba(X_test))

    # 3. ASSERT (La semilla fija garantiza predicciones idénticas)
    assert np.allclose(predicciones[0], predicciones[1])
