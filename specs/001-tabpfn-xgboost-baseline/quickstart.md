# Guía Rápida: Validación

Esta guía valida que la feature (TabPFN-v2 + baseline XGBoost) funciona end-to-end. Sigue los pasos en orden.

## Requisitos Previos

- Python 3.12, `uv` instalado
- Dataset: `data/raw/ai4i2020.csv` (10,000 registros)
- Dependencias: `uv sync` completado
- **Licencia de TabPFN aceptada** (una sola vez): registrarse en
  [ux.priorlabs.ai/account](https://ux.priorlabs.ai/account) y aceptar la licencia en
  [/account/licenses](https://ux.priorlabs.ai/account/licenses). Sin esto, TabPFN no
  descarga los pesos y sus pruebas se omiten automáticamente.

## Paso 1: Verificar Preprocesamiento

**Objetivo**: Confirmar que los datos se cargan y preparan correctamente (split estratificado, codificación one-hot).

```bash
uv run pytest tests/unit/test_load.py tests/unit/test_preprocess.py tests/unit/test_features.py -v
```

**Resultado esperado**: todas las pruebas pasan. Verifican 10.000 registros sin nulos,
one-hot de `Type` en `Type_H`/`Type_L`/`Type_M`, split 8000/2000 y proporción de fallos
de ~3,4% en ambos conjuntos.

## Paso 2: Verificar Integración de TabPFN-v2

**Objetivo**: Confirmar que TabPFN-v2 se carga y genera predicciones sin errores.

```bash
uv run pytest tests/unit/test_tabpfn.py tests/integration/test_tabpfn_e2e.py -v
```

**Resultado esperado**: las pruebas pasan y confirman probabilidades `(2000, 2)` en el
rango [0, 1] que suman 1 por fila, sin colapsar a una sola clase.

> Si aparecen como `SKIPPED`, la licencia de TabPFN aún no está aceptada (ver Requisitos Previos).

## Paso 3: Verificar Baseline XGBoost

**Objetivo**: Confirmar que XGBoost entrena y genera predicciones.

```bash
uv run pytest tests/unit/test_xgboost.py tests/integration/test_xgboost_e2e.py -v
```

**Resultado esperado**: las pruebas pasan. El `scale_pos_weight` calculado del desbalance
real es **28,52** y el recall sobre la clase falla es mayor que 0 (la matriz de confusión
no es trivial).

## Paso 4: Verificar Métricas de Evaluación

**Objetivo**: Confirmar que F1, Recall y PR-AUC se computan para ambos modelos.

```bash
uv run pytest tests/unit/test_metrics.py tests/integration/test_evaluation_e2e.py -v
```

**Resultado esperado**: las pruebas pasan y confirman que `compute_metrics()` devuelve
exactamente `f1`, `recall` y `pr_auc` —nunca accuracy— en el rango [0, 1].

## Paso 5: Verificar Logging en MLflow

**Objetivo**: Confirmar que ambos runs quedan registrados en MLflow.

```bash
uv run python -m scripts.evaluar_modelos
```

**Salida esperada**:

```text
Train: (8000, 8) | Test: (2000, 8) | Fallos en test: 68

Modelo             F1   Recall   PR-AUC
TabPFN-v2       0.825    0.765    0.880
XGBoost         0.729    0.750    0.837

Mejor recall sobre la clase falla: TabPFN-v2
```

Luego abre la interfaz de MLflow:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

En [http://localhost:5000](http://localhost:5000) debe existir el experimento
`001-tabpfn-xgboost-baseline` con los runs `tabpfn-v2` y `xgboost-baseline`, cada uno
con los tags `spec_id: 001` y `model_type`, y sus tres métricas.

> MLflow 3.16 dejó el almacenamiento en archivos (`./mlruns`) en modo mantenimiento, por
> eso el backend del proyecto es `sqlite:///mlflow.db` (ya ignorado por git).

## Resumen

Todos los pasos pasan ✅. Alternativamente, la suite completa se ejecuta con:

```bash
uv run pytest tests/ -q
```
