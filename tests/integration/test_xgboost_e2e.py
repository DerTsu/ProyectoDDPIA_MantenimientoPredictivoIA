"""T021 [US3] Test de integración: XGBoost sobre el dataset AI4I 2020 real."""

import numpy as np
import pytest

from src.models import XGBoostBaseline

UMBRAL = 0.5


def test_scale_pos_weight_refleja_el_desbalance(split_data):
    # 1. ARRANGE (Etiquetas reales de entrenamiento, con ~3,4% de fallos)
    _, y_train, _, _ = split_data

    # 2. ACT (Derivar el peso de la clase minoritaria desde los datos)
    modelo = XGBoostBaseline.from_class_balance(y_train)

    # 3. ASSERT (El peso equivale a negativos/positivos, en torno a 1:28)
    assert modelo.scale_pos_weight == pytest.approx((y_train == 0).sum() / (y_train == 1).sum())
    assert 20 < modelo.scale_pos_weight < 40


def test_shape_sobre_el_test_set(xgboost_proba):
    # 1. ARRANGE (Probabilidades del modelo entrenado sobre el test set)
    proba = xgboost_proba

    # 2. ACT (Leer la forma de la matriz)
    forma = proba.shape

    # 3. ASSERT (Una fila por registro de prueba y una columna por clase)
    assert forma == (2000, 2)


def test_probabilidades_validas(xgboost_proba):
    # 1. ARRANGE (Matriz de probabilidades del test set)
    proba = xgboost_proba

    # 2. ACT (Medir rango y suma por fila)
    minimo, maximo = proba.min(), proba.max()
    sumas = proba.sum(axis=1)

    # 3. ASSERT (Distribución de probabilidad bien formada)
    assert minimo >= 0.0 and maximo <= 1.0
    assert np.allclose(sumas, 1.0)


def test_matriz_de_confusion_no_es_trivial(xgboost_proba, split_data):
    # 1. ARRANGE (Probabilidades predichas y etiquetas reales del test set)
    _, _, _, y_test = split_data
    proba_falla = xgboost_proba[:, 1]

    # 2. ACT (Calcular el recall sobre la clase falla)
    y_pred = (proba_falla >= UMBRAL).astype(int)
    recall = int(((y_pred == 1) & (y_test == 1)).sum()) / y_test.sum()

    # 3. ASSERT (Con scale_pos_weight ajustado el modelo no ignora la clase minoría)
    assert recall > 0, "XGBoost colapsó: recall 0 para la clase falla"
