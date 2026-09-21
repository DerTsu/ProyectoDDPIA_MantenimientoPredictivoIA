# Feature Specification: EDA del Dataset AI4I 2020

**Feature Branch**: `feature/eda-jigg` (documentada aquí como `002-eda-preprocesamiento` según la constitución)

**Created**: 2026-09-14

**Status**: Implementado (documentado retroactivamente — el código se construyó antes que esta especificación). Alcance reducido respecto a su definición original — ver [Historial de Alcance](#historial-de-alcance-trazabilidad).

**Input**: Generar un análisis exploratorio de datos (EDA) del dataset AI4I 2020, sencillo, aislado del resto de la app, que produzca gráficas y tablas para construir un informe con conclusiones, sin modificar el pipeline de modelado ya existente en `src/`.

## User Scenarios & Testing

### User Story 1 - Data Scientist explora el dataset antes del modelado (Priority: P1)

Un integrante del equipo necesita entender la forma, el balance de clases y las relaciones entre variables del dataset AI4I 2020 antes de confiar en el pipeline de modelado, y necesita tablas y gráficas reutilizables para el documento formal de la entrega y para la sustentación.

**Why this priority**: Sin EDA, el equipo no tiene evidencia visual del desbalance de clases (~3.4%) ni de qué variables separan mejor los casos de falla — información necesaria para justificar decisiones ya tomadas en la constitución (Principio II: F1/Recall/PR-AUC en vez de accuracy).

**Independent Test**: Se ejecuta `uv run python EDA/eda.py` sobre `data/raw/ai4i2020.csv` y se verifica que se generan 4 tablas CSV y 5 gráficas PNG en `EDA/`, sin errores ni warnings de Ruff.

**Acceptance Scenarios**:

1. **Given** el dataset en `data/raw/ai4i2020.csv`, **When** se ejecuta `EDA/eda.py`, **Then** se cargan 10,000 filas y 14 columnas sin valores nulos.
2. **Given** el dataset cargado, **When** se calculan las tablas resumen, **Then** se generan `estadisticas_descriptivas.csv`, `balance_clases.csv`, `conteo_modos_falla.csv` y `conteo_tipo_producto.csv` en `EDA/`.
3. **Given** las tablas generadas, **When** se revisa `balance_clases.csv`, **Then** se confirma un desbalance de ~3.4% de la clase falla (339 de 10,000 registros).
4. **Given** el dataset cargado, **When** se generan las gráficas, **Then** se producen `balance_clases.png`, `modos_falla.png`, `distribuciones.png`, `boxplots_por_falla.png` y `correlaciones.png` en `EDA/figures/`.
5. **Given** las tablas y gráficas generadas, **When** se redacta `informe_eda.md`, **Then** el informe referencia las 5 imágenes y documenta conclusiones para el modelado (variables más informativas, correlaciones relevantes, recomendación de métricas).

---

### Edge Cases

- ¿Qué pasa si `data/raw/ai4i2020.csv` no existe? → `load_data()` lanza `FileNotFoundError` con un mensaje explícito, sin generar tablas ni gráficas parciales.
- ¿Qué pasa si el dataset tiene valores nulos? → El script los reporta en consola (`Valores nulos totales: N`), aunque no los limpia (fuera de alcance de esta feature; ver Assumptions).
- ¿Qué pasa si se vuelve a correr el script? → Sobrescribe tablas y gráficas existentes en `EDA/` de forma determinista (mismo dataset → mismo resultado).

## Requirements

### Functional Requirements

- **FR-001**: El sistema DEBE cargar el dataset AI4I 2020 desde `data/raw/ai4i2020.csv` y fallar explícitamente si el archivo no existe.
- **FR-002**: El sistema DEBE calcular y guardar estadísticas descriptivas (media, desviación estándar, min/max, cuartiles) de las 5 variables de proceso.
- **FR-003**: El sistema DEBE calcular y guardar el balance de clases (`Machine failure`) en conteo y porcentaje.
- **FR-004**: El sistema DEBE calcular y guardar el conteo por modo de falla (TWF, HDF, PWF, OSF, RNF) y por tipo de producto (L/M/H).
- **FR-005**: El sistema DEBE generar gráficas de balance de clases, modos de falla, distribuciones de variables de proceso, comparación de esas variables según ocurrencia de falla, y correlaciones entre ellas.
- **FR-006**: El sistema DEBE producir un informe en Markdown (`informe_eda.md`) que referencie las gráficas generadas y documente conclusiones orientadas al modelado.
- **FR-007**: El código DEBE cumplir el Principio VI de la constitución (docstrings Google, verificado con `ruff check`).

### Key Entities

- **Dataset**: AI4I 2020 (10,000 registros, 5 variables de proceso numéricas, 1 variable categórica `Type`, 1 target binario `Machine failure`, 5 columnas de modo de falla).
- **Tablas resumen**: 4 archivos CSV en `EDA/` con estadísticas, balance de clases, y conteos.
- **Figuras**: 5 archivos PNG en `EDA/figures/`.
- **Informe**: `EDA/informe_eda.md`, documento Markdown con las conclusiones para el equipo.

## Success Criteria

### Measurable Outcomes

- **SC-001**: `uv run python EDA/eda.py` corre sin errores ni excepciones sobre el dataset real.
- **SC-002**: `uv run ruff check EDA/` pasa sin errores (0 warnings).
- **SC-003**: Se generan exactamente 4 tablas CSV y 5 figuras PNG en cada ejecución.
- **SC-004**: El balance de clases reportado coincide con el desbalance documentado en la constitución (~3-4% clase falla).
- **SC-005**: `informe_eda.md` referencia las 5 figuras generadas y contiene al menos una sección de conclusiones para el modelado.

## Assumptions

- El dataset ya fue descargado manualmente y colocado en `data/raw/ai4i2020.csv` (no se distribuye en el repo, ver README).
- El dataset está limpio (sin valores faltantes), tal como asume también la constitución del proyecto.
- Esta feature es de solo lectura/análisis: no modifica `src/preprocessing/` ni el pipeline de modelado ya existente — es un insumo informativo adicional, aislado en `EDA/`.
- No requiere pruebas unitarias en `tests/` (Principio V) porque no es un módulo de `src/` con lógica de negocio reutilizable, sino un script de análisis exploratorio de un solo uso; se verifica manualmente y con Ruff en su lugar.

## Historial de Alcance (Trazabilidad)

Esta feature se definió con un alcance y se cerró con otro. El cambio se registra aquí para dejar trazabilidad de la decisión, en vez de corregir la definición original en silencio.

### Definición original — 2026-09-13, constitución v1.2.0

La **Matriz de Alcance** de `.specify/memory/constitution.md` reserva esta feature como:

> `specs/002-eda-preprocesamiento/` → Módulo 2, Sprint 2: EDA + limpieza (compañero)

Es decir, el identificador comprometía **dos** entregables: el análisis exploratorio **y** la limpieza/preprocesamiento del dataset. De ahí el sufijo `-preprocesamiento` en el nombre del directorio.

### Alcance final — 2026-09-18

La feature se cierra cubriendo **únicamente el EDA**. La mitad de preprocesamiento no se implementó aquí, y no queda como deuda pendiente, por dos razones verificadas:

1. **No hay limpieza que hacer.** El dataset AI4I 2020 no tiene valores faltantes. Lo confirman tres fuentes independientes: el informe de este EDA (`EDA/informe_eda.md`, sección 1), la constitución (*"Dataset AI4I 2020 está limpio (sin valores faltantes)"*), y `src/preprocessing/load.py`, que lanza `ValueError` si aparece cualquier nulo — condición que el test `tests/unit/test_load.py::test_sin_valores_faltantes` verifica en cada corrida.
2. **El preprocesamiento ya existe, en la spec 001.** `src/preprocessing/preprocess.py` (one-hot de `Type`, descarte de columnas fuera de alcance) y `src/preprocessing/features.py` (split estratificado 80/20) implementan el Principio III de la constitución. Duplicarlo aquí habría creado dos pipelines compitiendo por la misma responsabilidad.

### Consecuencia

- El nombre del directorio `002-eda-preprocesamiento` **se conserva** para no romper las referencias ya existentes en la constitución, en los tags `spec_id` de MLflow y en el historial de commits. Su alcance real, sin embargo, es solo EDA.
- **No queda trabajo pendiente** en esta feature por la parte de preprocesamiento.
- Cualquier trabajo futuro que sí modifique el pipeline (por ejemplo, features derivadas como potencia mecánica o ΔT a partir de las correlaciones documentadas en el informe) **no pertenece a esta feature**: requiere una spec propia, porque cambiaría `src/preprocessing/` y por tanto las métricas ya reportadas en el README y registradas en MLflow.
