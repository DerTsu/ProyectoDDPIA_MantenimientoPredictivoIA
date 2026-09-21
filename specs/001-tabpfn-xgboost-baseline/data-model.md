# Modelo de Datos & Entidades

## Entidad Dataset: AI4I 2020 Mantenimiento Predictivo

**Fuente**: UCI ML Repository (CC BY 4.0)
**Registros**: 10,000
**Limpio**: Sin valores faltantes

### Features Crudas (antes de codificación)

| Columna | Tipo | Rango | Notas |
|--------|------|-------|-------|
| `UDI` | int | 1-10000 | ID Único (índice) |
| `Product ID` | str | L47181, M14860, etc. | Serial, no usado en modelado |
| `Type` | cat | L, M, H | Tipo de producto → codificar one-hot |
| `Air temperature [K]` | float | ~298-300 | Kelvin, continua |
| `Process temperature [K]` | float | ~308-310 | Kelvin, continua |
| `Rotational speed [rpm]` | int | 1200-2900 | Continua |
| `Torque [Nm]` | float | 10-85 | Continua |
| `Tool wear [min]` | int | 0-250 | Continua, acumulativa |
| `Machine failure` | int | 0, 1 | **TARGET** (3% clase positiva) |
| `TWF` / `HDF` / `PWF` / `OSF` / `RNF` | int | 0, 1 | Submodos de falla (no usado hoy) |

### Features Procesadas (tras preprocesamiento)

| Columna | Tipo | Notas |
|--------|------|-------|
| `Type_H` | float (0/1) | One-hot desde `Type` |
| `Type_L` | float (0/1) | One-hot desde `Type` |
| `Type_M` | float (0/1) | One-hot desde `Type` (implícito si ambas arriba = 0) |
| `Air temperature [K]` | float | Normalizada o tal cual |
| `Process temperature [K]` | float | Normalizada o tal cual |
| `Rotational speed [rpm]` | float | Normalizada o tal cual |
| `Torque [Nm]` | float | Normalizada o tal cual |
| `Tool wear [min]` | float | Normalizada o tal cual |

**Salida**:
- **X_train**: (8000, 8) — 80% de datos, estratificado
- **y_train**: (8000,) — binario (0/1), ~3% positiva
- **X_test**: (2000, 8) — 20% de datos, estratificado
- **y_test**: (2000,) — binario (0/1), ~3% positiva

## Entidades de Modelo

### TabPFN-v2

**Tipo**: Transformer Preentrenado para Clasificación Tabular
**Entrada**: X_train (8000 × 8), y_train (8000,)
**Salida**: Probabilidades [0, 1] por clase por muestra (2000 × 2)
**Serialización**: `.safetensors` o `.ckpt` (formato PyTorch)
**Propiedad Clave**: Sin entrenamiento por gradiente, solo aprendizaje in-context

### Clasificador XGBoost

**Tipo**: Gradient Boosting
**Entrada**: X_train (8000 × 8), y_train (8000,) con `scale_pos_weight` = (neg_count / pos_count) ≈ 32
**Salida**: Probabilidades [0, 1] por clase por muestra (2000 × 2)
**Serialización**: `.json` (formato nativo XGBoost) o `.ubj` (binario)
**Propiedad Clave**: Entrenado desde cero, maneja desbalance mediante ponderación

## Entidad Métricas

| Métrica | Definición | Rango | Objetivo |
|--------|-----------|-------|--------|
| F1 (clase falla) | 2 × (Precisión × Recall) / (Precisión + Recall) | [0, 1] | > 0.3 (mejor que trivial) |
| Recall (clase falla) | TP / (TP + FN) | [0, 1] | > 0.2 (detectar al menos 20% de fallos reales) |
| PR-AUC | Área bajo curva Precisión-Recall | [0, 1] | > 0.3 (poder discriminativo) |

## Entidad Run de MLflow

**Campos por run**:
- `run_name`: "tabpfn-v2" o "xgboost-baseline"
- `model_type`: "tabpfn" o "xgboost"
- `spec_id`: "001"
- `metrics`: {f1, recall, pr_auc, ...}
- `params`: {scale_pos_weight, tree_depth, ...} (para XGBoost)
- `artifacts`: modelo serializado + CSV de predicciones

**Almacenamiento**: MLflow localhost:5000, guardado en `mlruns/` (git-ignorado)
