"""Cliente gRPC mínimo usado como `HEALTHCHECK` del contenedor `inference`.

Llama al propio `HealthCheck` del servicio (ver `mantenimiento.proto`) en vez
de solo verificar que el puerto esté abierto, para detectar también un modelo
que no llegó a cargar. Sale con código 0 si `ready=True`, 1 en cualquier otro
caso (timeout, error de conexión, `ready=False`).
"""

import os
import sys

import grpc

from proto import mantenimiento_pb2, mantenimiento_pb2_grpc

DEFAULT_ADDRESS = f"localhost:{os.environ.get('GRPC_PORT', '50051')}"
TIMEOUT_SEGUNDOS = 3.0


def main() -> int:
    """Consulta `HealthCheck` y traduce el resultado a un código de salida.

    Returns:
        `0` si el servicio respondió `ready=True`, `1` en cualquier otro caso.
    """
    try:
        with grpc.insecure_channel(DEFAULT_ADDRESS) as channel:
            stub = mantenimiento_pb2_grpc.InferenceServiceStub(channel)
            response = stub.HealthCheck(
                mantenimiento_pb2.HealthCheckRequest(), timeout=TIMEOUT_SEGUNDOS
            )
            return 0 if response.ready else 1
    except grpc.RpcError:
        return 1


if __name__ == "__main__":
    sys.exit(main())
