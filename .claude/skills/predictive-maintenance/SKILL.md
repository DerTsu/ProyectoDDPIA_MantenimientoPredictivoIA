---
name: predictive-maintenance
description: Activa con TabPFN-v2, AI4I 2020, falla, mantenimiento predictivo, Streamlit
---

# Mantenimiento Predictivo - UAO

**Modelo:** TabPFN-v2 transformer tabular, in-context (no fit por gradiente), límite <10k filas / <10 clases, CPU OK.
No proponer GridSearch, RandomizedSearch ni entrenamiento desde cero: el modelo viene preentrenado.

**Datos:** AI4I 2020 - 6 variables de proceso (Air Temperature, Process Temperature, Rotational Speed,
Torque, Tool Wear, Type L/M/H) → target `Machine failure` + modos TWF/HDF/PWF/OSF/RNF.
Desbalance de clase ~3.4%.

**Preprocesamiento:** one-hot de `Type`, split estratificado, StandardScaler solo para baselines
(TabPFN no lo requiere), SMOTE/class_weight opcionales.

**Métricas:** F1, recall y PR-AUC sobre la clase falla. Criterio de éxito Fase 1: superar al dummy.
Nunca usar accuracy como métrica principal (el desbalance la vuelve engañosa).

**Matriz de alcance:**
- INCLUIDO: transfer con TabPFN, preprocesamiento mínimo, baselines XGBoost/GradientBoosting, Streamlit, Docker, documentación.
- NO INCLUIDO: entrenar desde cero, series de tiempo, hardware IoT.

**CRISP-DM:** F1 negocio → datos → preparación (H1) → modelado TabPFN + XGBoost → evaluación →
despliegue Streamlit + Docker (H2, Módulo 3).

**Stack:** Python 3.12 (`pyproject.toml`), uv, scikit-learn/xgboost, Streamlit 1.63, Docker.
Dos servicios por gRPC (`proto/mantenimiento.proto`): `inference` (TabPFN-v2, puerto 50051)
y `web` (Streamlit, cliente gRPC, puerto 8501).

**Salida de `app/main.py`:** inputs de las variables de proceso + probabilidad + umbral ajustable +
semáforo verde/amarillo/rojo interpretable por personal no técnico.
