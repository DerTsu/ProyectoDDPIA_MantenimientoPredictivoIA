### 📝 Convención del Título de la PR

Aplica un [prefijo convencional](https://www.conventionalcommits.org/) en el título para clasificar el trabajo:

| Prefijo | Cuándo usar | Ejemplo |
|--------|-------------|---------|
| `feat:` | Nueva funcionalidad o módulo | `feat: integrar TabPFN-v2 como modelo base` |
| `fix:` | Corrección de errores o eliminación de warnings | `fix: corregir split estratificado con clase minoritaria` |
| `test:` | Adición o refactorización de pruebas | `test: agregar pruebas de métricas sobre la clase falla` |
| `docs:` | Especificaciones, constitución, README | `docs: enmendar constitución a v1.2.0` |
| `chore:` | Mantenimiento, dependencias, Docker, config | `chore: activar regla D de Ruff para docstrings` |

---

### 🧱 Módulo o Componente Afectado

Selecciona las áreas principales en las que trabaja esta PR:

- [ ] **Preprocesamiento** (`src/preprocessing/`: carga, one-hot, split estratificado)
- [ ] **Modelos** (`src/models/`: TabPFN-v2, baseline XGBoost)
- [ ] **Evaluación & Tracking** (`src/evaluation/`: métricas, runs de MLflow)
- [ ] **Scripts de ejecución** (`scripts/`)
- [ ] **Pruebas** (`tests/unit/`, `tests/integration/`)
- [ ] **Gobernanza Spec-Kit** (`.specify/memory/constitution.md`, `specs/`)
- [ ] **Interfaz Streamlit** (`app/`)
- [ ] **Infraestructura** (`pyproject.toml`, `.gitignore`, `Dockerfile`, pre-commit)
- [ ] **Documentación** (`README.md`, `docs/`)

---

### 📚 Descripción de los Cambios

<!--- Describe qué logra esta PR, la motivación y los detalles técnicos relevantes. -->

**Feature de Spec-Kit asociada:** <!-- Ej. specs/001-tabpfn-xgboost-baseline -->

#### Resumen de Cambios
-

#### Decisiones de Diseño
<!--- Explica las decisiones no obvias y, si te desviaste de lo planeado en tasks.md, por qué. -->
-

---

### ⚖️ Cumplimiento de la Constitución

Verifica que la PR respeta los principios vigentes (`.specify/memory/constitution.md`):

- [ ] **I — TabPFN-v2 sin gradiente:** no se entrena el modelo base ni se ajustan hiperparámetros por búsqueda.
- [ ] **II — Métricas de la clase falla:** se reportan F1, Recall y PR-AUC. **Nunca accuracy** como criterio de éxito.
- [ ] **III — Preprocesamiento mínimo y reemplazable:** la interfaz de `src/preprocessing/` sigue permitiendo insertar el EDA posterior.
- [ ] **IV — CRISP-DM:** el trabajo corresponde a una fase identificable y está documentado en su `spec.md`.
- [ ] **V — Tests con estructura AAA:** todos los tests usan `# 1. ARRANGE`, `# 2. ACT`, `# 3. ASSERT`.
- [ ] **VI — Docstrings Google:** toda función pública documenta `Args:`, `Returns:` y `Raises:`.
- [ ] Las desviaciones respecto a `tasks.md`, si las hay, están explicadas arriba.

---

### ✅ Lista de Chequeo Pre-PR

- [ ] **Entorno:** el código se ejecutó con **`uv`** y Python **3.12**.
- [ ] **Linting y formato:** `uv run ruff check .` y `uv run ruff format --check .` pasan limpios.
- [ ] **Pre-commit:** `uv run pre-commit run --all-files` pasa sin modificar archivos.
- [ ] **Pruebas:** `uv run pytest` pasa completo (indica abajo si hay pruebas omitidas y por qué).
- [ ] **Exclusiones (`.gitignore`):** el dataset (`data/raw/`), `mlruns/` y `mlflow.db` **NO** están rastreados por Git.
- [ ] **Estructura:** código en `src/`, scripts en `scripts/`, pruebas en `tests/`.
- [ ] **Reproducibilidad:** las semillas están fijadas (`random_state=42`) y el resultado se repite entre ejecuciones.
- [ ] **MLflow:** si se entrenó o evaluó un modelo, el run quedó registrado con los tags `spec_id` y `model_type`.

---

### 🧪 Evidencia de Pruebas Ejecutadas

<!--- Pega la salida real de los comandos, o adjunta capturas de la interfaz de MLflow. -->

```bash
uv run pytest
```

```text
# Pega aquí la salida
```

#### Métricas obtenidas (si aplica)

| Modelo | F1 | Recall | PR-AUC |
|--------|-----|--------|--------|
|        |     |        |        |
