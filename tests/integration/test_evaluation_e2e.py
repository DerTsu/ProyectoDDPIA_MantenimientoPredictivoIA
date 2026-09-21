"""T026 [US4] Test de integración: evaluación de ambos modelos y logging a MLflow."""

import mlflow
import pytest

from src.evaluation import compute_metrics, log_run_to_mlflow
from src.models import XGBoostBaseline


def test_metricas_de_xgboost_son_interpretables(xgboost_proba, split_data):
    # 1. ARRANGE (Predicciones del baseline y etiquetas reales)
    _, _, _, y_test = split_data
    proba_falla = xgboost_proba[:, 1]

    # 2. ACT (Calcular las métricas de la clase falla)
    metricas = compute_metrics(y_test, proba_falla)

    # 3. ASSERT (Valores útiles y dentro de rango)
    assert metricas["f1"] > 0.0
    assert metricas["recall"] > 0.0
    assert all(0.0 <= valor <= 1.0 for valor in metricas.values())


def test_metricas_de_tabpfn_son_interpretables(tabpfn_proba, split_data):
    # 1. ARRANGE (Predicciones del modelo base y etiquetas reales)
    _, _, _, y_test = split_data
    proba_falla = tabpfn_proba[:, 1]

    # 2. ACT (Calcular las métricas de la clase falla)
    metricas = compute_metrics(y_test, proba_falla)

    # 3. ASSERT (Valores útiles y dentro de rango)
    assert metricas["f1"] > 0.0
    assert metricas["recall"] > 0.0
    assert all(0.0 <= valor <= 1.0 for valor in metricas.values())


def test_comparacion_lado_a_lado(tabpfn_proba, xgboost_proba, split_data):
    # 1. ARRANGE (Predicciones de ambos modelos sobre el mismo test set)
    _, _, _, y_test = split_data

    # 2. ACT (Calcular el recall de cada uno y elegir el mejor)
    recall_tabpfn = compute_metrics(y_test, tabpfn_proba[:, 1])["recall"]
    recall_xgboost = compute_metrics(y_test, xgboost_proba[:, 1])["recall"]
    mejor = "tabpfn" if recall_tabpfn >= recall_xgboost else "xgboost"

    # 3. ASSERT (La comparación permite decidir cuál detecta más fallos)
    assert mejor in {"tabpfn", "xgboost"}


def test_logging_registra_run_con_tags_y_metricas(xgboost_proba, split_data, tmp_path):
    # 1. ARRANGE (Predicciones del baseline y un backend MLflow aislado)
    _, y_train, _, y_test = split_data
    tracking_uri = f"sqlite:///{tmp_path.as_posix()}/mlflow.db"

    # 2. ACT (Registrar el run del baseline)
    metricas = log_run_to_mlflow(
        model_type="xgboost",
        y_true=y_test,
        y_proba_pos=xgboost_proba[:, 1],
        params={"scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum()},
        experiment="test-001",
        tracking_uri=tracking_uri,
    )

    # 3. ASSERT (El run queda trazado a la spec 001 con sus tres métricas)
    run = mlflow.search_runs(experiment_names=["test-001"]).iloc[0]
    assert run["tags.spec_id"] == "001"
    assert run["tags.model_type"] == "xgboost"
    assert run["metrics.f1"] == pytest.approx(metricas["f1"])
    assert run["metrics.recall"] == pytest.approx(metricas["recall"])
    assert run["metrics.pr_auc"] == pytest.approx(metricas["pr_auc"])


def test_logging_registra_artefactos_y_tags_de_trazabilidad(split_data, xgboost_proba, tmp_path):
    # 1. ARRANGE (Backend MLflow aislado, modelo ajustado y un CSV que hace de dataset)
    X_train, y_train, _, y_test = split_data
    tracking_uri = f"sqlite:///{tmp_path.as_posix()}/mlflow.db"
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("columna\n1\n", encoding="utf-8")
    modelo = XGBoostBaseline.from_class_balance(y_train).fit(X_train, y_train)

    # 2. ACT (Registrar el run pasando modelo, dataset y enlace al PR)
    log_run_to_mlflow(
        model_type="xgboost",
        y_true=y_test,
        y_proba_pos=xgboost_proba[:, 1],
        experiment="test-artefactos",
        tracking_uri=tracking_uri,
        model=modelo,
        dataset_path=str(dataset),
        pr_link="https://github.com/ejemplo/pull/1",
    )

    # 3. ASSERT (Tags de trazabilidad, params de reproducibilidad y los tres artefactos)
    run = mlflow.search_runs(experiment_names=["test-artefactos"]).iloc[0]
    assert run["tags.model_license"] == "Apache-2.0"
    assert run["tags.pr_link"] == "https://github.com/ejemplo/pull/1"
    assert run["tags.author"]
    assert run["tags.environment"]
    assert run["params.model_repo"] == "dmlc/xgboost"
    assert run["params.device"] in {"cpu", "cuda"}
    assert run["params.dataset_version"]
    artefactos = {item.path for item in mlflow.MlflowClient().list_artifacts(run["run_id"])}
    assert artefactos == {"ejemplos_prediccion.csv", "reporte_evaluacion.txt", "model"}
