# Feature Specification: TabPFN-v2 + XGBoost Baseline

**Feature Branch**: `001-tabpfn-xgboost-baseline`

**Created**: 2026-09-13

**Status**: Draft

**Input**: Integrar TabPFN-v2 para clasificar fallos de máquinas usando AI4I 2020 dataset, comparado con baseline XGBoost. Evaluar con F1, Recall, PR-AUC sobre clase falla. Manejo de desbalance (~3%) con class_weight.

## User Scenarios & Testing

### User Story 1 - Data Engineer Prepares Dataset (Priority: P1)

Data engineer carga el dataset AI4I 2020 (10,000 registros, 6 variables de proceso + target de falla), aplica preprocesamiento mínimo (one-hot de Type L/M/H, split estratificado 80-20, normalización si aplica) y verifica que el dataset está listo para modelado.

**Why this priority**: Sin datos preparados, no se pueden entrenar ni evaluar los modelos. Es la base de todo.

**Independent Test**: Se ejecuta `src/preprocessing/build_features()`, se obtienen X (features), y (target), y se verifica que el split estratificado preserva la proporción ~3% de fallos en train y test.

**Acceptance Scenarios**:

1. **Given** dataset AI4I 2020 en `data/raw/ai4i2020.csv`, **When** se llama `load_raw_data()`, **Then** retorna DataFrame con 10,000 filas y 7 columnas (6 features + target)
2. **Given** dataset cargado, **When** se llama `build_features()`, **Then** retorna X (10,000 x 7 después de one-hot) e y (10,000 x 1 binario), con split train=8000 / test=2000
3. **Given** split estratificado, **When** se calcula proporción de fallos en train y test, **Then** ambos son ~3% (±1%)

---

### User Story 2 - ML Engineer Integrates TabPFN-v2 (Priority: P1)

ML engineer integra el modelo TabPFN-v2 preentrenado (sin entrenamiento adicional) para hacer predicciones de falla sobre X. Carga el checkpoint, ejecuta predict_proba(), obtiene probabilidades de falla por muestra.

**Why this priority**: TabPFN-v2 es el modelo base definido en la constitución. Su integración es bloqueante para evaluación.

**Independent Test**: Se carga TabPFN-v2, se pasa X_test (2000 muestras), se obtienen probabilidades [0, 1] para la clase falla. Se verifica que las predicciones son razonables (no todos 0 o todos 1).

**Acceptance Scenarios**:

1. **Given** checkpoint TabPFN-v2 disponible, **When** se llama `TabPFNClassifier.fit_predict_proba(X_train, y_train, X_test)`, **Then** retorna matriz (2000 x 2) con probabilidades de cada clase
2. **Given** probabilidades de TabPFN, **When** se aplica umbral 0.5, **Then** se generan predicciones binarias coherentes (no todas positivas ni todas negativas)

---

### User Story 3 - ML Engineer Trains XGBoost Baseline (Priority: P1)

ML engineer entrena XGBoost como baseline clásico usando los mismos X_train, y_train. Ajusta class_weight para manejar el desbalance (~3% de fallos). Realiza predicciones en X_test.

**Why this priority**: Necesario para poder comparar TabPFN-v2 vs. baseline y justificar la selección del modelo.

**Independent Test**: Se entrena XGBoost con `scale_pos_weight` calculado del desbalance. Se obtienen predicciones y probabilidades en test. Se verifica que el modelo no colapsa a una clase (e.g., predice fallos en al menos 1-2% de muestras de test).

**Acceptance Scenarios**:

1. **Given** X_train, y_train con desbalance 3%, **When** se entrena XGBoost con `scale_pos_weight` apropiado, **Then** modelo converge sin errores
2. **Given** modelo entrenado, **When** se predice en X_test, **Then** matriz de confusión no es trivial (recall > 0 para clase falla)

---

### User Story 4 - Data Scientist Evaluates Both Models (Priority: P1)

Data scientist calcula F1, Recall y PR-AUC para TabPFN-v2 y XGBoost sobre X_test, y_test. Compara resultados lado a lado. Loguea ambos runs a MLflow con métricas y artefactos (modelos serializados).

**Why this priority**: Evaluación es el criterio de éxito de la feature. Sin métricas claras, no se puede validar el desempeño.

**Independent Test**: Se generan reportes con métricas F1, Recall, PR-AUC para ambos modelos. Se verifica que están en rango [0, 1] y son interpretables. Se confirma que MLflow registró 2 runs distintos.

**Acceptance Scenarios**:

1. **Given** predicciones TabPFN y XGBoost en test, **When** se calculan F1, Recall, PR-AUC, **Then** todos los valores están en [0, 1]
2. **Given** ambas evaluaciones, **When** se comparan, **Then** se puede identificar cuál modelo tiene mejor Recall (sensibilidad) para detectar fallos
3. **Given** runs entrenados, **When** se loguean a MLflow, **Then** se pueden visualizar en `http://localhost:5000`

---

### Edge Cases

- ¿Qué pasa si el split estratificado falla (dataset muy pequeño)? → Se usa split simple 80-20 como fallback, con advertencia
- ¿Qué pasa si TabPFN no converge? → Se reporta error explícito, no se continúa con evaluación
- ¿Qué pasa si XGBoost predice con recall 0? → Se reporta en logs, se sugiere ajustar class_weight
- ¿Qué pasa si dataset tiene valores faltantes no detectados? → Se reporta durante preprocesamiento; no continuar hasta resolverlo

## Requirements

### Functional Requirements

- **FR-001**: Sistema DEBE cargar dataset AI4I 2020 desde `data/raw/ai4i2020.csv` sin errores
- **FR-002**: Sistema DEBE aplicar one-hot encoding a la variable `Type` (L/M/H) y generar X normalizado (numpy array o DataFrame)
- **FR-003**: Sistema DEBE realizar split estratificado 80-20 preservando proporción de clase falla (~3%)
- **FR-004**: Sistema DEBE integrar TabPFN-v2 sin reentrenamiento y generar probabilidades de falla [0, 1]
- **FR-005**: Sistema DEBE entrenar XGBoost con `scale_pos_weight` ajustado para desbalance (class_weight equivalente)
- **FR-006**: Sistema DEBE calcular F1, Recall y PR-AUC sobre la clase falla (no accuracy)
- **FR-007**: Sistema DEBE serializar ambos modelos en formatos nativos (TabPFN: .ckpt/.safetensors, XGBoost: .json)
- **FR-008**: Sistema DEBE loguear ambos runs a MLflow con métricas, parámetros y artefactos (tag `spec-id: 001`)

### Key Entities

- **Dataset**: AI4I 2020 (10k registros, 6 features numéricas + 1 feature categórica `Type` + 1 target binario `Machine failure`)
- **Model (TabPFN-v2)**: Transformer preentrenado, in-context learning, sin gradiente, formato `.safetensors`
- **Model (XGBoost)**: Gradient boosting, entrenado desde cero, `scale_pos_weight` para desbalance, formato `.json`
- **Metrics**: F1 (macro/micro), Recall (recall de clase falla), PR-AUC (Precision-Recall AUC para clase falla)
- **Run (MLflow)**: Log de experimento con modelo, hiperparámetros, métricas y timestamp

## Success Criteria

### Measurable Outcomes

- **SC-001**: Dataset se carga sin errores y tiene exactamente 10,000 registros y 6 variables de proceso observables
- **SC-002**: Preprocesamiento genera X con 7-8 columnas (após one-hot de Type) e y binario (0/1 para falla/no-falla)
- **SC-003**: Split estratificado asegura que train y test ambos tienen ~3% de fallos (±1%)
- **SC-004**: TabPFN-v2 genera predicciones en menos de 5 segundos para 2000 muestras de test (CPU)
- **SC-005**: XGBoost se entrena en menos de 10 segundos en CPU
- **SC-006**: Ambos modelos producen F1 > 0.3 sobre la clase falla (mejor que dummy classifier trivial)
- **SC-007**: Recall de ambos modelos > 0.2 (capacidad de detectar al menos 20% de los fallos verdaderos)
- **SC-008**: Ambos runs están registrados en MLflow con tags `spec-id: 001`, `model_type: {tabpfn, xgboost}` y al menos 3 métricas cada uno

## Assumptions

- Dataset AI4I 2020 está limpio (sin valores faltantes), tal como se reporta en documentación UCI
- TabPFN-v2 versión 8.5.0+ está disponible en PyPI y funciona sin necesidad de reentrenamiento
- XGBoost 3.2.0+ está disponible y `scale_pos_weight` es el parámetro recomendado para desbalance
- Métrica PR-AUC puede calcularse directamente con `sklearn.metrics.auc()` sin necesidad de librerías especiales
- Serialización de checkpoints (TabPFN y XGBoost) se maneja con formatos nativos de cada librería sin conversiones complejas
- MLflow 3.16.0+ está corriendo en localhost:5000 y acepta logs de modelos sin autenticación
- CPU es suficiente para entrenar XGBoost y ejecutar TabPFN (sin GPU requerida)
- Split estratificado es posible con `StratifiedKFold` o `train_test_split(..., stratify=y)` de sklearn
