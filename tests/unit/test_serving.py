"""Tests unitarios del servicer gRPC de inferencia (src/serving/server.py).

Llaman a los métodos del `InferenceServicer` directamente (sin sockets ni
canal gRPC real) para verificar la lógica de predicción; la comunicación de
red en sí se cubre en tests/integration/test_grpc_e2e.py.
"""

import math

from app.utils.preprocessing import preprocess_input
from proto import mantenimiento_pb2


def test_predict_failure_devuelve_probabilidades_validas(inference_servicer):
    # 1. ARRANGE (vector de 8 features de una fila concreta)
    features = preprocess_input("M", 298.1, 308.6, 1551, 42.8, 7)[0].tolist()
    request = mantenimiento_pb2.PredictionRequest(features=features)

    # 2. ACT
    response = inference_servicer.PredictFailure(request, context=None)

    # 3. ASSERT (probabilidades válidas que suman 1)
    assert 0.0 <= response.proba_no_failure <= 1.0
    assert 0.0 <= response.proba_failure <= 1.0
    assert math.isclose(response.proba_no_failure + response.proba_failure, 1.0, abs_tol=1e-6)


def test_predict_failure_es_determinista_para_la_misma_entrada(inference_servicer):
    # 1. ARRANGE (mismo vector de features en dos llamadas)
    features = preprocess_input("L", 300.0, 310.0, 1500, 40.0, 100)[0].tolist()

    # 2. ACT
    respuesta_1 = inference_servicer.PredictFailure(
        mantenimiento_pb2.PredictionRequest(features=features), context=None
    )
    respuesta_2 = inference_servicer.PredictFailure(
        mantenimiento_pb2.PredictionRequest(features=features), context=None
    )

    # 3. ASSERT (TabPFN-v2 con contexto fijo es determinista, ver Principio I)
    assert respuesta_1.proba_failure == respuesta_2.proba_failure


def test_health_check_reporta_listo_cuando_el_modelo_ya_cargo(inference_servicer):
    # 1. ARRANGE
    request = mantenimiento_pb2.HealthCheckRequest()

    # 2. ACT
    response = inference_servicer.HealthCheck(request, context=None)

    # 3. ASSERT
    assert response.ready is True
