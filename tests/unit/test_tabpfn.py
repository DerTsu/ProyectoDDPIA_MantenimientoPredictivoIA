"""T015 [US2] Tests unitarios para el envoltorio de TabPFN-v2.

Usan una muestra sintética pequeña: TabPFN en CPU es costoso y aquí solo se
verifica el contrato de la interfaz, no la calidad de las predicciones.
"""

import numpy as np
import pytest
from tabpfn.errors import TabPFNLicenseError

from src.models import TabPFNClassifier

SIN_LICENCIA = "Licencia de TabPFN no aceptada: ver https://ux.priorlabs.ai/account/licenses"


@pytest.fixture(scope="module")
def datos_sinteticos():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(60, 8))
    y_train = (rng.random(60) < 0.3).astype(int)
    X_test = rng.normal(size=(10, 8))
    return X_train, y_train, X_test


@pytest.fixture(scope="module")
def proba(datos_sinteticos):
    X_train, y_train, X_test = datos_sinteticos
    try:
        return TabPFNClassifier().predict_proba(X_train, y_train, X_test)
    except TabPFNLicenseError:
        pytest.skip(SIN_LICENCIA)


@pytest.fixture(scope="module")
def modelo_con_contexto(datos_sinteticos):
    X_train, y_train, _ = datos_sinteticos
    try:
        return TabPFNClassifier().fit_context(X_train, y_train)
    except TabPFNLicenseError:
        pytest.skip(SIN_LICENCIA)


def test_se_instancia_sin_errores():
    # 1. ARRANGE (Sin preparación: se prueba el constructor por defecto)

    # 2. ACT (Instanciar el clasificador preentrenado)
    modelo = TabPFNClassifier()

    # 3. ASSERT (Queda configurado para CPU y con semilla fija)
    assert modelo.device == "cpu"
    assert modelo.random_state == 42


def test_shape_de_las_probabilidades(proba, datos_sinteticos):
    # 1. ARRANGE (Probabilidades ya inferidas y datos de prueba usados)
    _, _, X_test = datos_sinteticos

    # 2. ACT (Leer la forma de la matriz de probabilidades)
    forma = proba.shape

    # 3. ASSERT (Una fila por muestra y una columna por clase)
    assert forma == (len(X_test), 2)


def test_probabilidades_en_rango_valido(proba):
    # 1. ARRANGE (Matriz de probabilidades ya inferida)
    matriz = proba

    # 2. ACT (Obtener el mínimo y el máximo)
    minimo, maximo = matriz.min(), matriz.max()

    # 3. ASSERT (Toda probabilidad vive en [0, 1])
    assert minimo >= 0.0
    assert maximo <= 1.0


def test_probabilidades_suman_uno(proba):
    # 1. ARRANGE (Matriz de probabilidades ya inferida)
    matriz = proba

    # 2. ACT (Sumar las probabilidades de cada fila)
    sumas = matriz.sum(axis=1)

    # 3. ASSERT (Las dos clases son exhaustivas y mutuamente excluyentes)
    assert np.allclose(sumas, 1.0)


def test_es_reproducible(datos_sinteticos):
    # 1. ARRANGE (Mismos datos y misma semilla en dos instancias)
    X_train, y_train, X_test = datos_sinteticos

    # 2. ACT (Inferir dos veces por separado)
    try:
        primera = TabPFNClassifier(random_state=42).predict_proba(X_train, y_train, X_test)
        segunda = TabPFNClassifier(random_state=42).predict_proba(X_train, y_train, X_test)
    except TabPFNLicenseError:
        pytest.skip(SIN_LICENCIA)

    # 3. ASSERT (La semilla fija garantiza predicciones idénticas)
    assert np.allclose(primera, segunda)


def test_fit_context_retorna_self(datos_sinteticos):
    # 1. ARRANGE (Un clasificador nuevo y los datos sintéticos)
    X_train, y_train, _ = datos_sinteticos
    modelo = TabPFNClassifier()

    # 2. ACT (Fijar el contexto y capturar el valor de retorno)
    try:
        retorno = modelo.fit_context(X_train, y_train)
    except TabPFNLicenseError:
        pytest.skip(SIN_LICENCIA)

    # 3. ASSERT (Retorna la misma instancia para permitir encadenamiento)
    assert retorno is modelo


def test_predict_proba_con_contexto_es_equivalente_a_predict_proba(
    modelo_con_contexto, proba, datos_sinteticos
):
    # 1. ARRANGE (Modelo con contexto fijado y probabilidades ya inferidas con
    # la llamada de una sola etapa; ambas usan random_state=42 por defecto)
    _, _, X_test = datos_sinteticos

    # 2. ACT (Predecir usando el contexto ya fijado)
    try:
        resultado = modelo_con_contexto.predict_proba_con_contexto(X_test)
    except TabPFNLicenseError:
        pytest.skip(SIN_LICENCIA)

    # 3. ASSERT (Mismas predicciones que predict_proba en una sola llamada)
    np.testing.assert_allclose(resultado, proba)


def test_encadenado_de_contexto_y_prediccion_es_equivalente(proba, datos_sinteticos):
    # 1. ARRANGE (Datos sintéticos y clasificador nuevo)
    X_train, y_train, X_test = datos_sinteticos

    # 2. ACT (Fijar el contexto y predecir en la misma cadena)
    try:
        encadenado = (
            TabPFNClassifier().fit_context(X_train, y_train).predict_proba_con_contexto(X_test)
        )
    except TabPFNLicenseError:
        pytest.skip(SIN_LICENCIA)

    # 3. ASSERT (La cadena da el mismo resultado que la llamada unitaria)
    np.testing.assert_allclose(encadenado, proba)
