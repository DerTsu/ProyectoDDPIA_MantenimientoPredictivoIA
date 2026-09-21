# Tareas: EDA del Dataset AI4I 2020

**Entrada**: Especificación desde `specs/002-eda-preprocesamiento/spec.md`

**Prerequisitos**: plan.md (requerido), spec.md (requerido)

**Nota**: Estas tareas se documentan retroactivamente — la implementación se completó antes de escribir este archivo, siguiendo una conversación iterativa en vez del flujo specify → plan → tasks → implement.

## Formato: `[ID] [Story] Descripción`

---

## Phase 1: Setup

- [X] T001 Crear carpeta `EDA/` y `EDA/figures/` en la raíz del repo
- [X] T002 Verificar que `data/raw/ai4i2020.csv` existe (10,000 filas, 14 columnas, sin nulos)

## Phase 2: User Story 1 — Data Scientist explora el dataset antes del modelado (Priority: P1)

- [X] T003 [US1] Implementar `load_data()` en `EDA/eda.py`: carga el CSV, lanza `FileNotFoundError` explícito si no existe
- [X] T004 [US1] Implementar `save_summary_tables()`: genera `estadisticas_descriptivas.csv`, `balance_clases.csv`, `conteo_modos_falla.csv`, `conteo_tipo_producto.csv`
- [X] T005 [US1] Implementar `plot_class_balance()`: gráfica de balance de clases con conteo y porcentaje anotado
- [X] T006 [US1] Implementar `plot_failure_modes()`: gráfica de conteo por modo de falla (TWF/HDF/PWF/OSF/RNF)
- [X] T007 [US1] Implementar `plot_distributions()`: histogramas de las 5 variables de proceso
- [X] T008 [US1] Implementar `plot_boxplots_by_failure()`: comparación de variables de proceso entre registros con y sin falla
- [X] T009 [US1] Implementar `plot_correlation_heatmap()`: mapa de calor de correlaciones entre variables de proceso
- [X] T010 [US1] Implementar `main()` que ejecuta el pipeline completo y se puede correr con `uv run python EDA/eda.py`
- [X] T011 [US1] Redactar `informe_eda.md` con las tablas, las 5 figuras referenciadas, y conclusiones para el modelado

**Checkpoint**: Script corre de punta a punta sin errores, genera 4 tablas + 5 figuras + 1 informe

## Phase 3: Calidad y consistencia con el equipo

- [X] T012 Corregir 3 violaciones de Ruff (`B905`, `zip()` sin `strict=`) detectadas al fusionar con la configuración de Ruff de `main`
- [X] T013 Reorganizar de `notebooks/eda.py` + `docs/eda/` a una única carpeta `EDA/` (decisión del autor, ver plan.md)
- [X] T014 Resolver conflictos de merge con `origin/main` (`.pre-commit-config.yaml`, `pyproject.toml`, `uv.lock`) sin perder configuración de ninguna de las dos partes
- [X] T015 Crear `AGENTS.md` en la raíz, alineado con `.specify/memory/constitution.md`, para que Gentleman Guardian Angel tenga reglas de revisión definidas
- [X] T016 Documentar esta feature retroactivamente como `specs/002-eda-preprocesamiento/` (este archivo, `spec.md`, `plan.md`)

**Checkpoint**: Feature lista para PR (`feature/eda-jigg` → `main`)
