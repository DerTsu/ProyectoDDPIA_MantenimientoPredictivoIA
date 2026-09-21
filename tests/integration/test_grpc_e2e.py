"""Prueba end-to-end de la comunicación gRPC entre `web` e `inference`.

A diferencia de tests/unit/test_serving.py (llama al servicer directo, sin
red), aquí se levanta un servidor gRPC real en un socket local y se le habla
con el mismo stub que usa app/utils/model.py, para probar la serialización
protobuf y el transporte de red de punta a punta.
"""

from concurrent import futures

import grpc
import pytest

from app.utils.preprocessing import preprocess_input
from proto import mantenimiento_pb2, mantenimiento_pb2_grpc


@pytest.fixture
def grpc_server_address(inference_servicer):
    """Servidor gRPC real, en un hilo aparte, escuchando en un puerto libre."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    mantenimiento_pb2_grpc.add_InferenceServiceServicer_to_server(inference_servicer, server)
    port = server.add_insecure_port("localhost:0")
    server.start()

    yield f"localhost:{port}"

    server.stop(grace=None)


def test_predict_failure_a_traves_de_un_canal_grpc_real(grpc_server_address):
    # 1. ARRANGE (canal y stub apuntando al servidor real levantado arriba)
    features = preprocess_input("H", 298.1, 308.6, 1551, 42.8, 7)[0].tolist()
    with grpc.insecure_channel(grpc_server_address) as channel:
        stub = mantenimiento_pb2_grpc.InferenceServiceStub(channel)

        # 2. ACT
        response = stub.PredictFailure(
            mantenimiento_pb2.PredictionRequest(features=features), timeout=30
        )

    # 3. ASSERT
    assert 0.0 <= response.proba_failure <= 1.0


def test_health_check_a_traves_de_un_canal_grpc_real(grpc_server_address):
    # 1. ARRANGE
    with grpc.insecure_channel(grpc_server_address) as channel:
        stub = mantenimiento_pb2_grpc.InferenceServiceStub(channel)

        # 2. ACT
        response = stub.HealthCheck(mantenimiento_pb2.HealthCheckRequest(), timeout=10)

    # 3. ASSERT
    assert response.ready is True


def test_canal_a_un_puerto_sin_servidor_lanza_rpc_error():
    # 1. ARRANGE (puerto localhost sin ningún servidor escuchando)
    with grpc.insecure_channel("localhost:1") as channel:
        stub = mantenimiento_pb2_grpc.InferenceServiceStub(channel)

        # 2. ACT / 3. ASSERT (app/utils/model.py traduce esto a
        # InferenceServiceUnavailableError; aquí se verifica el error de base)
        with pytest.raises(grpc.RpcError):
            stub.HealthCheck(mantenimiento_pb2.HealthCheckRequest(), timeout=3)
