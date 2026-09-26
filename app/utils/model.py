"""Cliente gRPC del servicio de inferencia TabPFN-v2.

Hasta la introducción de gRPC, este módulo cargaba TabPFN-v2 en el propio
proceso de Streamlit (`@st.cache_resource`). Ahora el modelo (y su stack
pesado: torch, tabpfn) vive en el servicio `inference` (ver
`src/serving/server.py`); esta interfaz solo abre un canal gRPC hacia él y
cachea el *stub*, no el modelo. El vector de 8 features se sigue preparando
aquí mismo con `app.utils.preprocessing.preprocess_input` — ese límite del
servicio no cambió, solo cómo se invoca la predicción.
"""

import os

import grpc
import numpy as np
import streamlit as st

from proto import mantenimiento_pb2, mantenimiento_pb2_grpc

INFERENCE_SERVICE_ADDR = os.environ.get("INFERENCE_SERVICE_ADDR", "localhost:50051")
TIMEOUT_SEGUNDOS = 60.0


class InferenceServiceUnavailableError(RuntimeError):
    """Se lanza cuando el servicio de inferencia gRPC no responde."""


@st.cache_resource(show_spinner=False)
def get_inference_stub(
    address: str = INFERENCE_SERVICE_ADDR,
) -> mantenimiento_pb2_grpc.InferenceServiceStub:
    """Abre (y cachea) el canal gRPC hacia el servicio de inferencia.

    Abrir el canal no bloquea ni valida conectividad: gRPC conecta de forma
    perezosa en la primera llamada real (`PredictFailure`/`HealthCheck`).

    Args:
        address: Dirección `host:puerto` del servicio de inferencia.

    Returns:
        Stub listo para invocar los métodos del servicio.
    """
    channel = grpc.insecure_channel(address)
    return mantenimiento_pb2_grpc.InferenceServiceStub(channel)


def check_inference_health() -> bool:
    """Verifica si el servicio de inferencia está listo para predecir.

    Returns:
        `True` si el servicio respondió `ready=True`; `False` si no respondió
        a tiempo o devolvió `ready=False`.
    """
    try:
        stub = get_inference_stub(INFERENCE_SERVICE_ADDR)
        response = stub.HealthCheck(
            mantenimiento_pb2.HealthCheckRequest(), timeout=TIMEOUT_SEGUNDOS
        )
        return bool(response.ready)
    except grpc.RpcError:
        return False


def predict_failure(X_new: np.ndarray) -> np.ndarray:
    """Invoca el servicio de inferencia gRPC para un vector de 8 features.

    Args:
        X_new: Vector `(1, 8)` ya preprocesado (ver
            `app.utils.preprocessing.preprocess_input`).

    Returns:
        Matriz `(1, 2)` con `[P(no falla), P(falla)]` — misma forma que
        devolvía el modelo en proceso, para no tocar
        `app/components/results.py`.

    Raises:
        InferenceServiceUnavailableError: Si el servicio no responde (caído,
            inalcanzable o supera el timeout).
    """
    stub = get_inference_stub(INFERENCE_SERVICE_ADDR)
    request = mantenimiento_pb2.PredictionRequest(features=list(X_new[0]))
    try:
        response = stub.PredictFailure(request, timeout=TIMEOUT_SEGUNDOS)
    except grpc.RpcError as exc:
        detail = exc.details() if hasattr(exc, "details") else str(exc)
        raise InferenceServiceUnavailableError(
            f"El servicio de inferencia ({INFERENCE_SERVICE_ADDR}) no respondió: {detail}"
        ) from exc
    return np.array([[response.proba_no_failure, response.proba_failure]])
