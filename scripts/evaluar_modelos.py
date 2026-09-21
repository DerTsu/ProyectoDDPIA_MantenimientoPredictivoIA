"""Evalúa TabPFN-v2 y el baseline XGBoost y registra ambos runs en MLflow.

Se ejecuta como módulo para que la raíz del proyecto quede en el path de importación:
    uv run python -m scripts.evaluar_modelos
    uv run python -m scripts.evaluar_modelos --sin-tabpfn   # omite el modelo base
"""

import argparse

from src.evaluation import log_run_to_mlflow
from src.models import TabPFNClassifier, XGBoostBaseline
from src.preprocessing import build_features, load_raw_data

RUTA_DATASET = "data/raw/ai4i2020.csv"


def main() -> None:
    """Evalúa los modelos seleccionados e imprime la comparación lado a lado.

    Lee los argumentos de línea de comandos (`--dataset`, `--sin-tabpfn`), entrena
    o infiere según corresponda, registra cada run en MLflow e imprime la tabla de
    métricas junto al modelo con mejor recall.

    Returns:
        None. El resultado se comunica por salida estándar y por los runs de MLflow.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=RUTA_DATASET)
    parser.add_argument("--sin-tabpfn", action="store_true", help="Evalúa solo XGBoost")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = build_features(load_raw_data(args.dataset))
    print(f"Train: {X_train.shape} | Test: {X_test.shape} | Fallos en test: {y_test.sum()}")

    resultados = {}

    if not args.sin_tabpfn:
        tabpfn = TabPFNClassifier()
        proba = tabpfn.predict_proba(X_train, y_train, X_test)
        resultados["TabPFN-v2"] = log_run_to_mlflow(
            "tabpfn",
            y_test,
            proba[:, 1],
            model=tabpfn,
            dataset_path=args.dataset,
        )

    modelo = XGBoostBaseline.from_class_balance(y_train)
    modelo.fit(X_train, y_train)
    proba = modelo.predict_proba(X_test)
    resultados["XGBoost"] = log_run_to_mlflow(
        "xgboost",
        y_test,
        proba[:, 1],
        params={"scale_pos_weight": round(modelo.scale_pos_weight, 4)},
        model=modelo,
        dataset_path=args.dataset,
    )

    print(f"\n{'Modelo':<12} {'F1':>8} {'Recall':>8} {'PR-AUC':>8}")
    for nombre, m in resultados.items():
        print(f"{nombre:<12} {m['f1']:>8.3f} {m['recall']:>8.3f} {m['pr_auc']:>8.3f}")

    if len(resultados) > 1:
        mejor = max(resultados, key=lambda n: resultados[n]["recall"])
        print(f"\nMejor recall sobre la clase falla: {mejor}")


if __name__ == "__main__":
    main()
