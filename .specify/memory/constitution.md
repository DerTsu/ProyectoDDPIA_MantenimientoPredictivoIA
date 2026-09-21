# Mantenimiento Predictivo UAO — Constitución del Proyecto

## Core Principles

### I. TabPFN-v2 como Modelo Base (In-Context Learning)

TabPFN-v2 es un transformer preentrenado para clasificación tabular que **NO requiere entrenamiento por gradiente**. No se proponen GridSearch, RandomizedSearch ni ajuste de hiperparámetros via gradiente. El modelo ingresa datos en contexto y produce predicciones directas. Esta decisión es definitiva para mantener la simpleza del pipeline y garantizar reproducibilidad en CPU.

### II. Métricas Obligatorias: F1, Recall, PR-AUC (No Accuracy)

El dataset tiene desbalance ~3.4% en la clase "falla". **Nunca usar accuracy como métrica de éxito** — es engañosa con desbalance. Las métricas obligatorias son: **F1 sobre la clase falla, Recall (sensibilidad), y PR-AUC (Area under Precision-Recall curve)**. Cualquier baseline (XGBoost, GradientBoosting) se evalúa con estas mismas métricas.

### III. Preprocesamiento Mínimo y Reemplazable

El preprocesamiento de hoy es deliberadamente mínimo: one-hot encoding de la variable `Type` (L/M/H), split estratificado por clase, y StandardScaler solo para baselines que lo requieran (TabPFN no lo requiere). Esta interfaz permite que trabajos posteriores de EDA/limpieza se inserten sin romper el pipeline de modelado actual.

### IV. CRISP-DM como Marco de Fases

El proyecto sigue CRISP-DM (Fase 1: Negocio/Datos, Fase 2: Preparación, Fase 3: Modelado, Fase 4: Evaluación, Fase 5: Despliegue). Cada fase documenta sus entradas, salidas, decisiones y criterios de éxito. Spec-kit (constitution → specify → plan → tasks) goberna cada incremento dentro de este marco.

### V. Tests con Estructura AAA (Arrange, Act, Assert)

Todo test de pytest DEBE escribirse en tres bloques separados por línea en blanco, encabezados por los comentarios literales `# 1. ARRANGE (...)`, `# 2. ACT (...)` y `# 3. ASSERT (...)`. Aplica a tests unitarios y de integración.

**Única excepción permitida — tests de excepciones (`pytest.raises`)**: cuando lo que se verifica es que una llamada falle, la ejecución y la verificación ocurren en la misma sentencia `with pytest.raises(...)` y no pueden separarse en dos bloques. En ese caso se usa un único bloque encabezado por el comentario literal `# 2. ACT + 3. ASSERT (...)`:

```python
def test_archivo_inexistente_lanza_error(tmp_path):
    # 1. ARRANGE (Ruta a un archivo que no existe)
    ruta = tmp_path / "no_existe.csv"

    # 2. ACT + 3. ASSERT (Cargar debe fallar con un error explícito)
    with pytest.raises(FileNotFoundError):
        load_raw_data(ruta)
```

Esta excepción NO aplica a ningún otro caso: si el acto y la aserción pueden separarse, deben separarse.

```python
def test_ejemplo():
    # 1. ARRANGE (Preparar datos de entrada)
    y = np.array([0] * 970 + [1] * 30)

    # 2. ACT (Ejecutar la función bajo prueba)
    modelo = XGBoostBaseline.from_class_balance(y)

    # 3. ASSERT (Verificar el resultado esperado)
    assert modelo.scale_pos_weight == pytest.approx(970 / 30)
```

**Rationale**: los comentarios explícitos permiten a cualquier integrante del equipo distinguir de un vistazo qué se prepara, qué se ejecuta y qué se verifica, sin leer la implementación. Cuando los datos provienen de una fixture, el bloque ARRANGE es la línea que la desempaqueta; si la fixture ya cacheó una operación costosa (por ejemplo, la inferencia de un modelo), el bloque ACT es la transformación que realmente se está probando.

### VI. Docstrings con Formato Google (Args / Returns / Raises)

Toda función, método y clase pública DEBE documentarse con un docstring en formato Google: una línea de resumen en imperativo, y las secciones `Args:`, `Returns:` y `Raises:` que apliquen, describiendo cada parámetro con su forma o unidad cuando sea relevante.

```python
def predict_proba(self, X_train, y_train, X_test) -> np.ndarray:
    """Predice probabilidades de clase para X_test usando X_train como contexto.

    Args:
        X_train: Matriz de entrenamiento (n_train, n_features).
        y_train: Etiquetas binarias (n_train,).
        X_test: Matriz a predecir (n_test, n_features).

    Returns:
        Matriz (n_test, 2) con [P(no falla), P(falla)] por muestra.
    """
```

**Rationale**: las formas de las matrices y el significado de cada valor no son evidentes desde la firma; documentarlos evita errores de integración entre módulos y permite que el equipo use el código sin leer su implementación.

**Verificación**: se hace cumplir automáticamente con Ruff (`select = ["D"]`, `convention = "google"` en `pyproject.toml`), de modo que `uv run ruff check` falla si falta un docstring o una sección. Los tests quedan exentos de `Args:`/`Returns:` porque se documentan con la estructura AAA del Principio V.

## Matriz de Alcance

### INCLUIDO en el Proyecto Completo

- Transfer learning con TabPFN-v2 (in-context, sin reentrenamiento)
- Preprocesamiento mínimo (one-hot, split, normalizacion)
- Baseline clásico: XGBoost (Módulo 2). TabPFN-Mix queda como referencia futura opcional
- Interfaz Streamlit (Módulo 3)
- Contenedorización Docker (Módulo 3)
- Tracking de experimentos con MLflow
- Documentación del proyecto (README, constitución, specs)

### EXCLUIDO (Out of Scope)

- Entrenar TabPFN-v2 desde cero (preentrenado es el punto)
- Series de tiempo / forecasting (problema de clasificación solo)
- Hardware IoT / edge deployment (Módulo 3 define esto)
- Multi-target: los 5 modos de falla (TWF/HDF/PWF/OSF/RNF) como targets separados — queda para extensión futura
- EDA exhaustivo de dataset (la EDA formal la aporta el compañero en Módulo 2; hoy es mínimo)

## Stack Técnico Fijo

- **Python**: 3.12 (fijado en `pyproject.toml` y `.python-version`)
- **Gestor de dependencias**: `uv` (0.12.3+)
- **Librerías core**: `tabpfn>=8.5.0`, `xgboost>=3.2.0`, `scikit-learn>=1.9.0`, `pandas>=3.0.5`, `numpy>=2.4.6`
- **Visualización**: `matplotlib>=3.11.1`
- **ML Ops**: `mlflow>=3.16.0` (tracking de runs/métricas)
- **Interfaz**: `streamlit>=1.63.0` (Módulo 3)
- **Contenedor**: Docker (Módulo 3, puerto 8501)
- **Calidad**: Ruff v0.16.7, pre-commit hooks, pytest

### Justificación de Modelos

**TabPFN-v2 como Modelo Base (Obligatorio)**:
- Modelo fundacional pre-entrenado (in-context learning, sin fit por gradiente)
- Model Card disponible en Hugging Face
- Desempeño en datasets pequeños (<10k filas) respaldado por publicación científica (Nature, Hollmann et al., 2025)
- Serialización: checkpoints PyTorch (`.ckpt` / `.safetensors`)

**Baselines Clásicos (Opcionales, Módulo 2 extensión)**:
- **XGBoost**: Baseline obligatorio en Módulo 2 Sprint 1 (hoy). Entrenado desde cero. Serialización: `.json` / `.ubj`
- **Gradient Boosting (scikit-learn)**: Baseline opcional para extensión posterior. Entrenado desde cero. Serialización: `.pkl` / `.joblib`
- **TabPFN-Mix**: Opcional, referencia dentro de la familia TabPFN para contrastar variantes. Serialización: `.ckpt` / `.safetensors`

**Criterio**: TabPFN-v2 se elige como modelo base definitivo por ser pre-entrenado, científicamente respaldado y tener Model Card. Baselines se documentan como referentes clásicos contra los cuales validar desempeño de forma objetiva.

### Preprocesamiento y Manejo del Desbalance

- Dataset AI4I 2020 está limpio (sin valores faltantes)
- Transformaciones esenciales: normalización/escalado de variables numéricas, one-hot de `Type` (L/M/H), split estratificado
- Desbalance de clase (~3% falla): manejar con **class_weight** (automático en sklearn) o **SMOTE** (sobremuestreo sintético, opcional según task.md)
- Interfaz de preprocesamiento clara en `src/preprocessing/` para que EDA posterior se integre sin romper pipeline de modelos

## Desarrollo Dirigido por Especificación (Spec-Driven)

Cada incremento de trabajo se organiza en una **feature de spec-kit** bajo `specs/<NNN>-<nombre>/`:

- `specs/001-tabpfn-xgboost-baseline/` → Módulo 2, Sprint 1: TabPFN-v2 + XGBoost baseline ✅
- `specs/002-eda-preprocesamiento/` → Módulo 2, Sprint 2: EDA (compañero) ✅ — cerrada solo con el EDA; la limpieza no se implementó porque el dataset no tiene nulos y el preprocesamiento ya vive en la spec 001. Trazabilidad completa en la sección "Historial de Alcance" de su `spec.md`. El nombre del directorio se conserva para no romper referencias ni los tags `spec_id` de MLflow.
- `specs/003-interfaz-streamlit/` → Módulo 3: interfaz web con Streamlit ✅ — antes listada aquí como `003-app-streamlit-docker`. Se implementó solo la interfaz; el despliegue se separó a la feature 004 para no reabrir una spec ya cerrada y mergeada.
- `specs/004-despliegue-docker/` → Módulo 3: contenedorización y despliegue (puerto 8501)

El baseline GradientBoosting queda **descartado**: XGBoost ya cumple el rol de baseline clásico frente al cual contrastar TabPFN-v2, y añadir un segundo modelo de la misma familia no aportaría evidencia nueva.

Cada feature contiene su propio `spec.md` (qué, por qué, criterio de éxito), `plan.md` (cómo, arquitectura), y `tasks.md` (tareas ordenadas). La **constitution** es el contrato que rige **todas** las features.

## Governance

### Amendment Procedure

Cambios a esta constitución requieren:
1. Justificación (problema concreto, restricción nueva, descubrimiento del equipo)
2. Propuesta de enmienda (principio/sección modificada)
3. Aprobación por consenso del equipo (incluir al profesor si afecta rubrica)
4. Versionamiento semántico (ver abajo)
5. Commit con referencia a la constitución anterior

### Versioning Policy

- **MAJOR** (1.0.0 → 2.0.0): Cambio incompatible en principios o matriz de alcance que invalida specs previas o decisiones pasadas
- **MINOR** (1.0.0 → 1.1.0): Nuevo principio, nueva sección, o expansión significativa de guidance sin romper lo anterior
- **PATCH** (1.0.0 → 1.0.1): Clarificaciones, fixes de typos, reordenamientos, ejemplos mejorados

### Compliance Review

Toda feature (spec-kit `specify` → `tasks`) DEBE verificar:
1. ¿Respeta la constitución en cada task? (métricas, stack, scope, etc.)
2. ¿Están explícitas las desviaciones si las hay?
3. ¿Se loguean decisiones en el spec.md o comentarios de código?
4. ¿Los tests siguen la estructura AAA del Principio V?
5. ¿Todas las funciones públicas tienen docstring Google según el Principio VI? (`ruff check` lo valida)

MLflow es el registro oficial de runs; cada run DEBE indicar a qué spec-kit feature pertenece (tag `spec-id: 001`, etc.).

---

**Version**: 1.2.1 | **Ratified**: 2026-09-13 | **Last Amended**: 2026-09-18

**Changelog v1.2.1 (PATCH)**: (1) Principio V — se documenta la excepción del bloque `# 2. ACT + 3. ASSERT` para tests con `pytest.raises`, formalizando una práctica ya presente y consistente en 4 tests del repositorio. (2) Desarrollo Dirigido por Especificación — se corrige el nombre de la feature 003 (`003-interfaz-streamlit`, no `003-app-streamlit-docker`), se registra el alcance final de la 002 y se lista la nueva feature `004-despliegue-docker`. Ningún principio cambia de contenido; solo se aclaran y se alinean con la realidad del repositorio.
