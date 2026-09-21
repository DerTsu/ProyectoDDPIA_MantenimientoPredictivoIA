"""T012 [US1] Test de integración: flujo completo de preprocesamiento."""

import numpy as np

from src.preprocessing import build_features, load_raw_data


def test_flujo_completo_load_a_features(dataset_path):
    # 1. ARRANGE (Ruta al dataset crudo, sin ningún paso previo)
    ruta = dataset_path

    # 2. ACT (Encadenar carga y construcción de features)
    X_train, y_train, X_test, y_test = build_features(load_raw_data(ruta))

    # 3. ASSERT (El pipeline completo entrega los cuatro conjuntos esperados)
    assert X_train.shape == (8000, 8)
    assert X_test.shape == (2000, 8)
    assert len(y_train) == 8000
    assert len(y_test) == 2000


def test_test_set_tiene_fallos_suficientes_para_evaluar(split_data):
    # 1. ARRANGE (Etiquetas del conjunto de prueba)
    _, _, _, y_test = split_data

    # 2. ACT (Contar los fallos disponibles para medir)
    n_fallos = int(y_test.sum())

    # 3. ASSERT (Con menos de 50 las métricas de la clase minoría no son estables)
    assert n_fallos >= 50, f"Solo {n_fallos} fallos en test"


def test_features_sin_nan_ni_infinitos(split_data):
    # 1. ARRANGE (Matrices de features de ambos conjuntos)
    X_train, _, X_test, _ = split_data

    # 2. ACT (Buscar valores no finitos en ambas matrices)
    hay_no_finitos = any(not np.isfinite(matriz).all() for matriz in (X_train, X_test))

    # 3. ASSERT (Los modelos requieren entradas numéricas finitas)
    assert not hay_no_finitos
