"""T025 [US4] Tests unitarios para compute_metrics()."""

import numpy as np
import pytest

from src.evaluation import compute_metrics

METRICAS_ESPERADAS = {"f1", "recall", "pr_auc"}


def test_retorna_las_tres_metricas_obligatorias():
    # 1. ARRANGE (Etiquetas reales y probabilidades predichas)
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Exactamente F1, Recall y PR-AUC)
    assert set(metricas) == METRICAS_ESPERADAS


def test_no_incluye_accuracy():
    # 1. ARRANGE (Un caso mínimo cualquiera)
    y_true = np.array([0, 1])
    y_proba = np.array([0.2, 0.8])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (La constitución prohíbe accuracy con clase desbalanceada)
    assert "accuracy" not in metricas


def test_prediccion_perfecta():
    # 1. ARRANGE (Probabilidades que separan perfectamente ambas clases)
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.1, 0.9, 1.0])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Las tres métricas alcanzan su valor máximo)
    assert metricas["f1"] == pytest.approx(1.0)
    assert metricas["recall"] == pytest.approx(1.0)
    assert metricas["pr_auc"] == pytest.approx(1.0)


def test_modelo_que_nunca_predice_falla():
    # 1. ARRANGE (Probabilidades siempre por debajo del umbral)
    y_true = np.array([0, 0, 0, 1])
    y_proba = np.array([0.01, 0.02, 0.03, 0.04])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Un modelo que nunca alerta no tiene mérito, pese a acertar el 75%)
    assert metricas["recall"] == 0.0
    assert metricas["f1"] == 0.0


def test_recall_se_calcula_sobre_la_clase_falla():
    # 1. ARRANGE (Dos fallos reales, de los que el modelo solo detecta uno)
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.1, 0.9, 0.2])

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Detecta 1 de 2 fallos)
    assert metricas["recall"] == pytest.approx(0.5)


def test_todas_las_metricas_en_rango_cero_uno():
    # 1. ARRANGE (Datos aleatorios con desbalance similar al real)
    rng = np.random.default_rng(7)
    y_true = (rng.random(200) < 0.05).astype(int)
    y_proba = rng.random(200)

    # 2. ACT (Calcular las métricas)
    metricas = compute_metrics(y_true, y_proba)

    # 3. ASSERT (Ninguna métrica se sale de [0, 1])
    assert all(0.0 <= valor <= 1.0 for valor in metricas.values())


def test_umbral_configurable():
    # 1. ARRANGE (Un fallo real con probabilidad predicha de 0.4)
    y_true = np.array([0, 1])
    y_proba = np.array([0.3, 0.4])

    # 2. ACT (Evaluar con el umbral por defecto y con uno más sensible)
    con_umbral_alto = compute_metrics(y_true, y_proba, threshold=0.5)
    con_umbral_bajo = compute_metrics(y_true, y_proba, threshold=0.35)

    # 3. ASSERT (Bajar el umbral recupera el fallo que se escapaba)
    assert con_umbral_alto["recall"] == 0.0
    assert con_umbral_bajo["recall"] == 1.0
