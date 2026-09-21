"""T016 [US2] Test de integración: TabPFN-v2 sobre el dataset AI4I 2020 real."""

import numpy as np

UMBRAL = 0.5


def test_shape_sobre_el_test_set(tabpfn_proba):
    # 1. ARRANGE (Probabilidades inferidas sobre los 2000 registros de prueba)
    proba = tabpfn_proba

    # 2. ACT (Leer la forma de la matriz)
    forma = proba.shape

    # 3. ASSERT (Una fila por registro de prueba y una columna por clase)
    assert forma == (2000, 2)


def test_probabilidades_validas(tabpfn_proba):
    # 1. ARRANGE (Matriz de probabilidades del test set)
    proba = tabpfn_proba

    # 2. ACT (Medir rango y suma por fila)
    minimo, maximo = proba.min(), proba.max()
    sumas = proba.sum(axis=1)

    # 3. ASSERT (Distribución de probabilidad bien formada)
    assert minimo >= 0.0 and maximo <= 1.0
    assert np.allclose(sumas, 1.0)


def test_predicciones_no_son_triviales(tabpfn_proba):
    # 1. ARRANGE (Probabilidades de la clase falla)
    proba_falla = tabpfn_proba[:, 1]

    # 2. ACT (Binarizar con el umbral de decisión y contar alertas)
    n_positivos = int((proba_falla >= UMBRAL).sum())

    # 3. ASSERT (El modelo no colapsa a una sola clase)
    assert 0 < n_positivos < len(proba_falla), f"Predicción degenerada: {n_positivos} positivos"


def test_detecta_parte_de_los_fallos_reales(tabpfn_proba, split_data):
    # 1. ARRANGE (Probabilidades predichas y etiquetas reales del test set)
    _, _, _, y_test = split_data
    proba_falla = tabpfn_proba[:, 1]

    # 2. ACT (Contar los fallos reales que el modelo sí alertó)
    y_pred = (proba_falla >= UMBRAL).astype(int)
    verdaderos_positivos = int(((y_pred == 1) & (y_test == 1)).sum())

    # 3. ASSERT (El modelo base debe aportar algo de sensibilidad)
    assert verdaderos_positivos > 0, "TabPFN no detectó ningún fallo real"
