"""Registro de runs en MLflow.

MLflow es el registro oficial de experimentos del proyecto: cada run debe indicar
a qué feature de spec-kit pertenece mediante el tag `spec_id` (Governance de la
constitución, FR-008).
"""

import getpass
import hashlib
import os
import platform
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.evaluation.metrics import UMBRAL_POR_DEFECTO, compute_metrics

EXPERIMENTO_POR_DEFECTO = "001-tabpfn-xgboost-baseline"
SPEC_ID = "001"

# MLflow 3.16 dejó el file store (`./mlruns`) en modo mantenimiento y exige un
# backend de base de datos para las funciones actuales.
TRACKING_URI_POR_DEFECTO = "sqlite:///mlflow.db"

NOMBRES_DE_RUN = {
    "tabpfn": "tabpfn-v2",
    "xgboost": "xgboost-baseline",
}

REPO_MODELO = {
    "tabpfn": "priorlabs/tabpfn-v2",
    "xgboost": "dmlc/xgboost",
}

LICENCIA_MODELO = {
    "tabpfn": "Prior Labs License (uso no comercial / académico)",
    "xgboost": "Apache-2.0",
}

# Cada modelo se empaca con su propio `save()`, en formato nativo (FR-007).
EXTENSION_MODELO = {
    "tabpfn": ".tabpfn",
    "xgboost": ".json",
}

N_EJEMPLOS_PREDICCION = 10


def _detectar_device() -> str:
    """Detecta el backend de cómputo disponible.

    Returns:
        `"cuda"` si hay GPU disponible vía torch, o `"cpu"` en caso contrario
        (incluyendo cuando torch no está instalado).
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _hash_dataset(dataset_path: str | None) -> str | None:
    """Calcula un hash corto del archivo de dataset para trazabilidad.

    Args:
        dataset_path: Ruta al archivo del dataset. Puede ser `None`.

    Returns:
        Los primeros 12 caracteres del hash MD5 del archivo, o `None` si
        `dataset_path` es `None` o el archivo no existe.
    """
    if not dataset_path or not Path(dataset_path).exists():
        return None
    return hashlib.md5(Path(dataset_path).read_bytes()).hexdigest()[:12]


def log_run_to_mlflow(
    model_type: str,
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    params: dict | None = None,
    spec_id: str = SPEC_ID,
    experiment: str = EXPERIMENTO_POR_DEFECTO,
    threshold: float = UMBRAL_POR_DEFECTO,
    tracking_uri: str | None = None,
    model: object | None = None,
    dataset_path: str | None = None,
    pr_link: str | None = None,
) -> dict[str, float]:
    """Registra un run con sus tags, parámetros, métricas y artefactos.

    Args:
        model_type: `"tabpfn"` o `"xgboost"`; define el nombre del run y el tag.
        y_true: Etiquetas reales del conjunto de prueba.
        y_proba_pos: Probabilidad predicha de la clase falla.
        params: Hiperparámetros adicionales a registrar (p. ej. `scale_pos_weight`).
        spec_id: Feature de spec-kit a la que pertenece el run.
        experiment: Nombre del experimento MLflow.
        threshold: Umbral de decisión usado en F1 y Recall.
        tracking_uri: Backend de MLflow. Por defecto toma `MLFLOW_TRACKING_URI`
            y, si no está definida, `sqlite:///mlflow.db`.
        model: Modelo ya ajustado, para empacarlo como artefacto con su propio
            método `save()`. Si es `None` no se registra el modelo.
        dataset_path: Ruta al CSV usado, para registrar `dataset_version`.
        pr_link: URL del PR o Issue asociado al run. Por defecto toma
            `MLFLOW_PR_LINK`; si ninguna está definida, el tag se omite.

    Returns:
        Las métricas registradas.
    """
    metricas = compute_metrics(y_true, y_proba_pos, threshold=threshold)

    mlflow.set_tracking_uri(
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or TRACKING_URI_POR_DEFECTO
    )
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=NOMBRES_DE_RUN.get(model_type, model_type)):
        mlflow.set_tag("spec_id", spec_id)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("model_license", LICENCIA_MODELO.get(model_type, "desconocida"))
        mlflow.set_tag("author", os.environ.get("MLFLOW_RUN_AUTHOR") or getpass.getuser())
        mlflow.set_tag("environment", f"{platform.system()}-{platform.release()}")
        link = pr_link or os.environ.get("MLFLOW_PR_LINK")
        if link:
            mlflow.set_tag("pr_link", link)

        params_completos = dict(params or {})
        params_completos["threshold"] = threshold
        params_completos["model_repo"] = REPO_MODELO.get(model_type, model_type)
        params_completos["device"] = _detectar_device()
        version_dataset = _hash_dataset(dataset_path)
        if version_dataset:
            params_completos["dataset_version"] = version_dataset
        mlflow.log_params(params_completos)

        mlflow.log_metrics(metricas)
        _log_artefactos(model_type, y_true, y_proba_pos, metricas, threshold, model)

    return metricas


def _log_artefactos(
    model_type: str,
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    metricas: dict[str, float],
    threshold: float,
    model: object | None,
) -> None:
    """Registra en el run activo los artefactos exigidos por el seguimiento.

    Escribe en un directorio temporal las predicciones de ejemplo, el reporte de
    evaluación y, si se recibe un modelo, su archivo en formato nativo; luego los
    sube al run de MLflow que esté abierto.

    Args:
        model_type: `"tabpfn"` o `"xgboost"`; define la extensión del modelo.
        y_true: Etiquetas reales del conjunto de prueba.
        y_proba_pos: Probabilidad predicha de la clase falla.
        metricas: Métricas ya calculadas, para el reporte de evaluación.
        threshold: Umbral de decisión con el que se derivan las predicciones.
        model: Modelo ajustado con método `save()`, o `None` para omitirlo.

    Returns:
        None. El efecto queda en los artefactos del run activo de MLflow.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        ejemplos = pd.DataFrame(
            {
                "y_true": np.asarray(y_true)[:N_EJEMPLOS_PREDICCION],
                "y_proba_falla": np.asarray(y_proba_pos)[:N_EJEMPLOS_PREDICCION],
            }
        )
        ejemplos["y_pred"] = (ejemplos["y_proba_falla"] >= threshold).astype(int)
        ruta_ejemplos = tmp_path / "ejemplos_prediccion.csv"
        ejemplos.to_csv(ruta_ejemplos, index=False)
        mlflow.log_artifact(str(ruta_ejemplos))

        lineas = [
            f"Reporte de evaluación - {NOMBRES_DE_RUN.get(model_type, model_type)}",
            "=" * 60,
            f"Muestras de prueba: {len(np.asarray(y_true))}",
            f"Fallas reales en prueba: {int(np.asarray(y_true).sum())}",
            f"Umbral de decisión: {threshold}",
            "",
            "Métricas sobre la clase falla:",
        ]
        lineas += [f"  {nombre}: {valor:.4f}" for nombre, valor in sorted(metricas.items())]
        ruta_reporte = tmp_path / "reporte_evaluacion.txt"
        ruta_reporte.write_text("\n".join(lineas) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(ruta_reporte))

        if model is None:
            return
        ruta_modelo = tmp_path / f"modelo_{model_type}{EXTENSION_MODELO.get(model_type, '.bin')}"
        try:
            model.save(ruta_modelo)
        except Exception as error:
            print(f"Advertencia: no se pudo empacar el modelo {model_type} ({error})")
        else:
            mlflow.log_artifact(str(ruta_modelo), artifact_path="model")
