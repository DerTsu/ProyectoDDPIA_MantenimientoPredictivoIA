# Tareas: TabPFN-v2 + Baseline XGBoost

**Entrada**: Especificación desde `specs/001-tabpfn-xgboost-baseline/spec.md`

**Prerequisitos**: plan.md (requerido), spec.md (requerido para user stories), data-model.md, quickstart.md

**Organización**: Tareas agrupadas por user story para permitir implementación e testing independientes de cada historia.

## Formato: `[ID] [P?] [Story] Descripción`

- **[P]**: Puede ejecutarse en paralelo (archivos diferentes, sin dependencias)
- **[Story]**: Qué user story pertenece esta tarea (e.g., US1, US2, US3, US4)
- Incluir rutas exactas de archivos en descripciones

---

## Phase 1: Setup (Infraestructura Compartida)

**Propósito**: Inicialización del proyecto y estructura básica

- [X] T001 Crear estructura de directorios `src/preprocessing/`, `src/models/`, `src/evaluation/` según plan.md
- [X] T002 Crear archivos `__init__.py` en cada módulo de `src/`
- [X] T003 [P] Crear `tests/` subdirectorios: `tests/unit/`, `tests/integration/`
- [X] T004 Verificar `data/raw/ai4i2020.csv` existe con 10,000 registros (no modificar, solo verificar)

---

## Phase 2: Foundational (Prerequisites Bloqueantes)

**Propósito**: Infraestructura bloqueante que DEBE completarse antes de cualquier user story

**⚠️ CRÍTICO**: Ninguna historia de usuario puede comenzar hasta que esta fase esté completa

- [X] T005 Crear `src/preprocessing/__init__.py` con importaciones de funciones públicas
- [X] T006 Implementar `load_raw_data(filepath: str) -> pd.DataFrame` en `src/preprocessing/load.py`:
  - Leer CSV desde `data/raw/ai4i2020.csv`
  - Retornar DataFrame con columnas: UDI, Product ID, Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure
  - Validar que tiene exactamente 10,000 filas
  - Validar que no hay valores faltantes (asserción según assumption de constitution)
  - Documentar en docstring la función de signature y retorno
- [X] T007 Implementar `preprocess_features(df: pd.DataFrame) -> pd.DataFrame` en `src/preprocessing/preprocess.py`:
  - Aplicar one-hot encoding a columna `Type` (L/M/H) → Type_H, Type_L, Type_M (3 dummies, según data-model.md)
  - Dropear columnas originales (`UDI`, `Product ID`, `Type`, los 5 submodos TWF/HDF/PWF/OSF/RNF)
  - Retornar DataFrame con 9 columnas: Type_H, Type_L, Type_M, Air temp, Process temp, RPM, Torque, Tool wear, Machine failure
  - Documentar restricciones de features en docstring
- [X] T008 Implementar `build_features(df: pd.DataFrame) -> tuple(X_train, y_train, X_test, y_test)` en `src/preprocessing/features.py`:
  - Llamar internamente `preprocess_features(df)` para preparar datos
  - Hacer split estratificado 80-20 usando `train_test_split(..., stratify=y)` de sklearn
  - X: numpy array de shape (n_samples, 8) sin la columna target
  - y: numpy array de shape (n_samples,) binaria (0/1)
  - Retornar tupla: (X_train 8000×8, y_train 8000, X_test 2000×8, y_test 2000)
  - Verificación en docstring: "Train/test class ratio debe ser ~3% ± 1%"
  - Usar `random_state=42` en split para reproducibilidad

**Checkpoint**: Foundational listo — implementación de user stories puede comenzar en paralelo

---

## Phase 3: User Story 1 — Data Engineer Prepares Dataset (Priority: P1)

**Objetivo**: Cargar y preparar dataset AI4I 2020 con split estratificado y validación de desbalance

**Independent Test**: Ejecutar `src/preprocessing/build_features()`, verificar X_train/y_train shapes, verificar ~3% de fallos en ambos sets

### Tests para User Story 1 (OPCIONAL - solo si se solicita TDD)

> **NOTA**: Estos tests DEBEN FALLAR antes de implementación en Phase 2

- [X] T009 [P] [US1] Test unitario para `load_raw_data()` en `tests/unit/test_load.py`:
  - Verificar que retorna DataFrame con shape (10000, 7)
  - Verificar que todas las columnas esperadas existen
  - Verificar que no hay valores faltantes (NaN)
- [X] T010 [P] [US1] Test unitario para `preprocess_features()` en `tests/unit/test_preprocess.py`:
  - Verificar que one-hot de Type genera Type_H y Type_L
  - Verificar shape resultante (10000, 8)
  - Verificar que columnas originales `Type` y `Product ID` se dropean
- [X] T011 [P] [US1] Test unitario para `build_features()` en `tests/unit/test_features.py`:
  - Verificar shapes de X_train, y_train, X_test, y_test
  - Verificar que proporciones de clase son ~3% en train y test (±1%)
  - Verificar que X es numpy array, y es binaria (0/1)
- [X] T012 [US1] Test de integración para preprocesamiento end-to-end en `tests/integration/test_preprocessing_e2e.py`:
  - Ejecutar flujo completo: `load_raw_data()` → `build_features()`
  - Verificar que `y_test` tiene al menos 50 fallos (para poder calcular métricas razonables)

### Implementación para User Story 1

Las funciones T006, T007, T008 YA están en Phase 2 (Foundational) — esta es la verificación

- [X] T013 [US1] Crear `src/preprocessing/__init__.py` si aún no existe con importaciones públicas
- [X] T014 [US1] Ejecutar validación de preprocessing manualmente (resultado: X_train (8000, 8), fallos train 3.39%, test 3.40%):
  ```bash
  cd <project-root>
  uv run python -c "from src.preprocessing import load_raw_data, build_features; df = load_raw_data('data/raw/ai4i2020.csv'); X_train, y_train, X_test, y_test = build_features(df); print(f'X_train: {X_train.shape}, y_train pos rate: {y_train.sum()/len(y_train)*100:.1f}%')"
  ```
  - Verificar que no hay errores
  - Verificar que shapes son correctos (8000×8, 2000×8)
  - Verificar que proportions de clase están en rango esperado

**Checkpoint**: User Story 1 completada — dataset preparado y validado, listo para modelos

---

## Phase 4: User Story 2 — ML Engineer Integrates TabPFN-v2 (Priority: P1)

**Objetivo**: Integrar modelo TabPFN-v2 preentrenado sin entrenamiento adicional, generar predicciones en X_test

**Independent Test**: Cargar TabPFN-v2, pasar X_test (2000 muestras), obtener probabilidades [0,1], verificar coherencia (no todas positivas/negativas)

### Tests para User Story 2 (OPCIONAL)

- [X] T015 [P] [US2] Test unitario para `TabPFNClassifier` en `tests/unit/test_tabpfn.py`:
  - Verificar que clase carga sin errores
  - Verificar que `predict_proba()` retorna shape (n_samples, 2)
  - Verificar que probabilities están en rango [0, 1] y suman a 1 por fila
- [X] T016 [US2] Test de integración para TabPFN con datos reales en `tests/integration/test_tabpfn_e2e.py`:
  - Cargar datos con `build_features()`
  - Llamar `TabPFNClassifier.predict_proba(X_train, y_train, X_test)`
  - Verificar que predicciones no son triviales (no todas clase 0 o todas clase 1)

### Implementación para User Story 2

- [X] T017 [P] [US2] Implementar clase `TabPFNClassifier` en `src/models/tabpfn_model.py`:
  - Importar TabPFNClassifier desde librería `tabpfn`
  - Crear wrapper que encapsula el modelo preentrenado
  - Método `predict_proba(X_train, y_train, X_test) -> np.ndarray`:
    - X_train: (8000, 8), y_train: (8000,)
    - X_test: (2000, 8)
    - Retorna: (2000, 2) probabilidades [P(class=0), P(class=1)]
    - Usar `random_state=42` para reproducibilidad
    - Documentar en docstring: "In-context learning, no gradient training"
- [X] T018 [US2] Crear `src/models/__init__.py` con importación de `TabPFNClassifier`
- [X] T019 [US2] Ejecutar validación manual de TabPFN. Requirió aceptar la licencia de
  Prior Labs (https://ux.priorlabs.ai/accept-license?hf_repo_id=tabpfn_3) y habilitar
  `ignore_pretraining_limits=True`, porque TabPFN bloquea por defecto CPU con >5000
  muestras y X_train tiene 8000. Resultado: predicciones (2000, 2) válidas.
  ```bash
  cd <project-root>
  uv run python -c "from src.preprocessing import load_raw_data, build_features; from src.models import TabPFNClassifier; df = load_raw_data('data/raw/ai4i2020.csv'); X_train, y_train, X_test, y_test = build_features(df); clf = TabPFNClassifier(); proba = clf.predict_proba(X_train, y_train, X_test); print(f'Predictions shape: {proba.shape}, min: {proba.min():.3f}, max: {proba.max():.3f}')"
  ```
  - Verificar shape (2000, 2)
  - Verificar que probabilities están en [0, 1]

**Checkpoint**: User Story 2 completada — TabPFN-v2 integrado e integrando datos sin errores

---

## Phase 5: User Story 3 — ML Engineer Trains XGBoost Baseline (Priority: P1)

**Objetivo**: Entrenar XGBoost con `scale_pos_weight` ajustado para desbalance, generar predicciones

**Independent Test**: Entrenar XGBoost, obtener predicciones en test, verificar que matriz de confusión no es trivial (recall > 0)

### Tests para User Story 3 (OPCIONAL)

- [X] T020 [P] [US3] Test unitario para `XGBoostBaseline` en `tests/unit/test_xgboost.py`:
  - Verificar que clase se instancia sin errores
  - Verificar que `.fit()` completa sin errores
  - Verificar que `predict_proba()` retorna shape (n_samples, 2) y valores en [0, 1]
- [X] T021 [US3] Test de integración para XGBoost con datos reales en `tests/integration/test_xgboost_e2e.py`:
  - Cargar datos con `build_features()`
  - Calcular `scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()`
  - Entrenar XGBoost y obtener predicciones
  - Verificar que recall de clase falla > 0

### Implementación para User Story 3

- [X] T022 [P] [US3] Implementar clase `XGBoostBaseline` en `src/models/xgboost_baseline.py` (nombre propio para no ensombrecer `xgboost.XGBClassifier`):
  - Wrappear `xgboost.XGBClassifier` de librería `xgboost`
  - Método `__init__()`: aceptar `scale_pos_weight`, `random_state=42`
  - Método `fit(X_train, y_train)`: entrenar modelo
  - Método `predict_proba(X_test) -> np.ndarray`: retornar (n_samples, 2) probabilidades
  - Usar `scale_pos_weight` calculado como (neg_count / pos_count) ≈ 32 para manejar ~3% desbalance
  - Documentar en docstring: "Gradient boosting entrenado desde cero, maneja desbalance con scale_pos_weight"
- [X] T023 [US3] Crear entrada en `src/models/__init__.py` para importar `XGBoostBaseline`
- [X] T024 [US3] Ejecutar validación manual de XGBoost (resultado: `scale_pos_weight` = 28.52, predicciones (2000, 2)):
  ```bash
  cd <project-root>
  uv run python -c "from src.preprocessing import load_raw_data, build_features; from src.models import XGBClassifier; df = load_raw_data('data/raw/ai4i2020.csv'); X_train, y_train, X_test, y_test = build_features(df); scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum(); clf = XGBClassifier(scale_pos_weight=scale_pos_weight); clf.fit(X_train, y_train); proba = clf.predict_proba(X_test); print(f'Predictions shape: {proba.shape}')"
  ```
  - Verificar que entrenamiento completa sin errores
  - Verificar shape (2000, 2)

**Checkpoint**: User Story 3 completada — XGBoost entrenado sin errores

---

## Phase 6: User Story 4 — Data Scientist Evaluates Both Models (Priority: P1)

**Objetivo**: Calcular F1, Recall, PR-AUC para ambos modelos, loguear a MLflow, comparar resultados

**Independent Test**: Generar reportes con F1/Recall/PR-AUC para ambos modelos, verificar valores en [0, 1], verificar MLflow registró 2 runs

### Tests para User Story 4 (OPCIONAL)

- [X] T025 [P] [US4] Test unitario para `compute_metrics()` en `tests/unit/test_metrics.py`:
  - Verificar que calcula F1, Recall, PR-AUC correctamente
  - Verificar que valores están en rango [0, 1]
  - Verificar que PR-AUC se calcula solo para clase positiva
- [X] T026 [US4] Test de integración para evaluación completa en `tests/integration/test_evaluation_e2e.py`:
  - Ejecutar preprocesamiento → TabPFN → XGBoost → métricas
  - Verificar que ambos modelos producen F1 > 0.0 y Recall > 0.0
  - Verificar comparación lado a lado

### Implementación para User Story 4

- [X] T027 [P] [US4] Implementar función `compute_metrics(y_true, y_proba_pos, threshold=0.5)` en `src/evaluation/metrics.py` (se sustituyó `class_idx` por `threshold`: la columna de la clase positiva ya se pasa como argumento):
  - y_true: array binario (0/1) de shape (n_samples,)
  - y_proba_pos: array de probabilidades para clase positiva, shape (n_samples,)
  - Retornar dict: `{'f1': float, 'recall': float, 'pr_auc': float}`
  - F1: usar `sklearn.metrics.f1_score(y_true, y_pred_binary)` con umbral 0.5
  - Recall: usar `sklearn.metrics.recall_score(y_true, y_pred_binary)`
  - PR-AUC: usar `sklearn.metrics.auc(precision, recall)` donde precision/recall vienen de `precision_recall_curve(y_true, y_proba_pos)`
  - Documentar en docstring: "Todas métricas se calculan para la clase positiva (falla), nunca accuracy"
- [X] T028 [US4] Crear `src/evaluation/__init__.py` con importación de `compute_metrics`
- [X] T029 [P] [US4] Implementar logging a MLflow en `src/evaluation/tracking.py` (se evitó el nombre `logging.py` para no ensombrecer el módulo estándar):
  - Función `log_run_to_mlflow(model_type: str, y_test, y_proba, params: dict, spec_id='001')`:
    - `model_type`: "tabpfn" o "xgboost"
    - Calcular métricas usando `compute_metrics()`
    - Crear/activar experiment `001-tabpfn-xgboost-baseline`
    - Loguear con tags: `spec_id: 001`, `model_type: {tabpfn|xgboost}`
    - Loguear parámetros (ej: scale_pos_weight para XGBoost)
    - Loguear métricas: f1, recall, pr_auc
  - Usar `mlflow.set_experiment()` y `mlflow.start_run()` context manager
  - Documentar en docstring
- [X] T030 [US4] Ejecutar validación manual de evaluación mediante `scripts/evaluar_modelos.py`. Resultado: TabPFN-v2 F1 0.825 / Recall 0.765 / PR-AUC 0.880 vs. XGBoost F1 0.729 / Recall 0.750 / PR-AUC 0.837. TabPFN-v2 gana en las tres métricas:
  ```bash
  cd <project-root>
  uv run python -c "
  import mlflow
  from src.preprocessing import load_raw_data, build_features
  from src.models import TabPFNClassifier, XGBClassifier
  from src.evaluation import compute_metrics

  df = load_raw_data('data/raw/ai4i2020.csv')
  X_train, y_train, X_test, y_test = build_features(df)

  # TabPFN
  clf_tabpfn = TabPFNClassifier()
  y_proba_tabpfn = clf_tabpfn.predict_proba(X_train, y_train, X_test)
  metrics_tabpfn = compute_metrics(y_test, y_proba_tabpfn[:, 1])
  print(f'TabPFN metrics: {metrics_tabpfn}')

  # XGBoost
  scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
  clf_xgb = XGBClassifier(scale_pos_weight=scale_pos_weight)
  clf_xgb.fit(X_train, y_train)
  y_proba_xgb = clf_xgb.predict_proba(X_test)
  metrics_xgb = compute_metrics(y_test, y_proba_xgb[:, 1])
  print(f'XGBoost metrics: {metrics_xgb}')

  print(f'\nComparación: TabPFN F1={metrics_tabpfn[\"f1\"]:.3f} vs XGBoost F1={metrics_xgb[\"f1\"]:.3f}')
  "
  ```
  - Verificar que ambos modelos producen métricas en rango [0, 1]
  - Verificar que se puede identificar cuál tiene mejor recall
- [X] T031 [US4] Ejecutar MLflow logging y verificar en UI. Backend: `sqlite:///mlflow.db` (MLflow 3.16 dejó `./mlruns` en modo mantenimiento). Runs `tabpfn-v2` y `xgboost-baseline` registrados con sus tags y métricas:
  ```bash
  cd <project-root>
  uv run mlflow ui --port 5000 &
  # Esperar 2 segundos
  uv run python -c "
  import mlflow
  from src.preprocessing import load_raw_data, build_features
  from src.models import TabPFNClassifier, XGBClassifier
  from src.evaluation import compute_metrics, log_run_to_mlflow

  df = load_raw_data('data/raw/ai4i2020.csv')
  X_train, y_train, X_test, y_test = build_features(df)

  # TabPFN run
  clf_tabpfn = TabPFNClassifier()
  y_proba_tabpfn = clf_tabpfn.predict_proba(X_train, y_train, X_test)
  log_run_to_mlflow('tabpfn', y_test, y_proba_tabpfn[:, 1], params={}, spec_id='001')

  # XGBoost run
  scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
  clf_xgb = XGBClassifier(scale_pos_weight=scale_pos_weight)
  clf_xgb.fit(X_train, y_train)
  y_proba_xgb = clf_xgb.predict_proba(X_test)
  log_run_to_mlflow('xgboost', y_test, y_proba_xgb[:, 1], params={'scale_pos_weight': scale_pos_weight}, spec_id='001')

  print('Visita http://localhost:5000 para ver los 2 runs loguados')
  "
  ```
  - Abrir http://localhost:5000 en navegador
  - Verificar que experiment `001-tabpfn-xgboost-baseline` existe
  - Verificar que hay 2 runs: `tabpfn-v2` y `xgboost-baseline`
  - Verificar que cada run tiene tags `spec_id: 001` y `model_type`
  - Verificar que cada run muestra f1, recall, pr_auc

**Checkpoint**: User Story 4 completada — ambos modelos evaluados, comparados, loguados a MLflow

---

## Phase 7: Polish & Cross-Cutting Concerns

**Propósito**: Mejoras que afectan múltiples user stories y validación final

- [X] T032 [P] Ejecutar `uv run ruff check .` y resolver cualquier issue de linting
- [X] T033 [P] Ejecutar `uv run ruff format .` para formatear código
- [X] T034 [P] Ejecutar todos los tests con `uv run pytest tests/` (53 passed, 0 skipped)
- [X] T035 Ejecutar validación end-to-end del quickstart.md (los 5 pasos verificados):
  - Correr todos los 5 pasos del quickstart (Paso 1-5)
  - Verificar que cada paso produce output esperado
  - Documentar cualquier issue o ajuste necesario en quickstart.md
- [X] T036 Documentar decisiones de implementación en `src/` (docstrings, comentarios en código)
- [X] T037 Crear o actualizar `README.md` con sección de "Resultados de la Feature 001"
  - Incluir resumen de F1/Recall/PR-AUC de ambos modelos
  - Incluir link a MLflow runs: `http://localhost:5000`

---

## Dependencias & Orden de Ejecución

### Phase Dependencies

- **Setup (Phase 1)**: Sin dependencias — puede comenzar inmediatamente
- **Foundational (Phase 2)**: Depende de Setup completado — **BLOQUEA todas las user stories**
- **User Stories (Phase 3-6)**: Todas dependen de Foundational completado
  - User Story 1 (Phase 3): Sin dependencias en otras historias
  - User Story 2 (Phase 4): Puede comenzar después de US1 completa (depende de datos preparados)
  - User Story 3 (Phase 5): Puede comenzar después de US1 completa (depende de datos preparados)
  - User Story 4 (Phase 6): Depende de US2 Y US3 completadas (necesita predicciones de ambos)
- **Polish (Phase 7)**: Depende de todas las user stories deseadas completadas

### User Story Dependencies

- **US1 (Preprocesamiento)**: Sin dependencias — puede empezar tras Foundational
- **US2 (TabPFN-v2)**: Depende de US1 (necesita X_train, y_train, X_test, y_test)
- **US3 (XGBoost)**: Depende de US1 (necesita X_train, y_train, X_test, y_test)
- **US4 (Evaluación)**: Depende de US2 AND US3 (necesita predicciones de ambos)

### Oportunidades de Paralelismo

- Todos los tasks T001-T004 de Setup marked [P] pueden ejecutarse en paralelo
- Todos los tasks de Foundational (T005-T008) se ejecutan secuencialmente (dependencias lógicas)
- **Una vez Foundational completa**:
  - US2 (T015-T019) puede ejecutarse en paralelo con US3 (T020-T024)
  - Pero ambas dependen de US1 completa
  - US4 (T025-T031) espera a US2 Y US3
- **Ejecución recomendada para 1 persona**: Secuencial: Setup → Foundational → US1 → [US2, US3 en paralelo lógico] → US4 → Polish

---

## Estrategia de Implementación

### MVP First (Solo User Story 1 + 2 + 3 + 4)

1. Completar Phase 1: Setup
2. Completar Phase 2: Foundational (CRÍTICO — bloquea todo)
3. Completar Phase 3: User Story 1 (preprocesamiento)
4. Completar Phase 4: User Story 2 (TabPFN-v2)
5. Completar Phase 5: User Story 3 (XGBoost)
6. Completar Phase 6: User Story 4 (evaluación + MLflow)
7. **STOP y VALIDAR**: Ejecutar quickstart.md end-to-end
8. Desplegar / demo si está listo

### Entrega Incremental

Todas las 4 user stories son P1 y deben completarse hoy (Módulo 2, Sprint 1), así que no hay "MVP parcial". La entrega es el conjunto completo.

---

## Notas Finales

- **Formato de tests obligatorio: AAA (Arrange, Act, Assert)** con los comentarios literales
  `# 1. ARRANGE (...)`, `# 2. ACT (...)` y `# 3. ASSERT (...)` separando los tres bloques.
  Aplica a todos los tests, unitarios y de integración — ver Principio V de la constitución.
- [P] tasks = archivos diferentes, sin dependencias → pueden ejecutarse en paralelo
- [Story] label mapea cada tarea a una user story específica para trazabilidad
- Cada user story es independientemente completable y testeable
- Pre-commit hooks (Ruff) se ejecutarán automáticamente en `git add` — resolver antes de commit
- Todos los tests son OPCIONALES en esta versión (solo si se solicita TDD)
- Constitution (principios, stack, scope) debe verificarse en cada task
- MLflow es el registro oficial de runs; cada run DEBE indicar spec_id: 001
- Quickstart.md es la guía de validación end-to-end — ejecutar antes de considerar feature completa
