# Plan de Implementación: EDA del Dataset AI4I 2020

**Rama**: `feature/eda-jigg` | **Fecha**: 2026-09-14 | **Spec**: `specs/002-eda-preprocesamiento/spec.md`

**Entrada**: Especificación de feature desde `specs/002-eda-preprocesamiento/spec.md`

## Resumen

Construir un script de análisis exploratorio de datos (EDA) sobre el dataset AI4I 2020, aislado del pipeline de modelado (`src/`), que genere tablas y gráficas descriptivas y un informe en Markdown con conclusiones orientadas al modelado (desbalance de clases, variables más informativas, correlaciones).

## Contexto Técnico

**Lenguaje/Versión**: Python 3.12 (uv)

**Dependencias Principales** (ya declaradas en `pyproject.toml`, no se agregó ninguna nueva):
- `pandas>=3.0.5` (carga y agregación de datos)
- `matplotlib>=3.11.1` (gráficas)
- `numpy>=2.4.6` (transitiva de pandas)

**Almacenamiento**: Archivos locales
- Entrada: `data/raw/ai4i2020.csv` (10,000 filas × 14 columnas), compartida con `src/preprocessing/`
- Salida: `EDA/*.csv` (tablas), `EDA/figures/*.png` (gráficas), `EDA/informe_eda.md` (informe)

**Testing**: Verificación manual + `ruff check` (ver Assumptions en spec.md sobre por qué no aplica pytest aquí)

**Plataforma Objetivo**: CPU local (sin GPU, sin dependencias pesadas)

**Tipo de Proyecto**: Script de análisis exploratorio de un solo módulo, sin API ni clases

**Objetivos de Rendimiento**: Ejecución completa (carga + 4 tablas + 5 gráficas) en menos de 10 segundos sobre 10,000 registros

**Restricciones**:
- No debe modificar ni depender de `src/preprocessing/` (evita conflictos con el pipeline de modelado de otro integrante)
- No debe requerir dependencias nuevas fuera de las ya declaradas en `pyproject.toml`

**Escala/Alcance**: Dataset de 10k registros, 5 variables de proceso, salida de 4 tablas + 5 figuras + 1 informe

## Verificación de Constitución

✅ **Principio I (TabPFN-v2 sin ajuste por gradiente)**: N/A — esta feature no entrena ni usa modelos.
✅ **Principio II (F1, Recall, PR-AUC, no accuracy)**: PASA. El EDA documenta explícitamente el desbalance (~3.4%) y refuerza en el informe por qué no debe usarse accuracy.
✅ **Principio III (Preprocesamiento mínimo y reemplazable)**: PASA. El EDA es de solo lectura, no toca `src/preprocessing/`; se integra sin romper el pipeline existente, tal como previó la constitución.
✅ **Principio IV (CRISP-DM)**: PASA. Corresponde a la Fase 2 (Preparación de datos / Entendimiento de datos) de CRISP-DM.
⚠️ **Principio V (Tests con estructura AAA)**: Desviación justificada — ver Complexity Tracking.
✅ **Principio VI (Docstrings Google)**: PASA. Todas las funciones públicas tienen docstring Google; `ruff check EDA/` pasa sin errores.
✅ **Matriz de Alcance**: PASA. La constitución reserva explícitamente `specs/002-eda-preprocesamiento/` para "EDA + limpieza (compañero)".

## Project Structure

### Documentation (this feature)

```text
specs/002-eda-preprocesamiento/
├── plan.md              # Este archivo
├── spec.md              # Especificación de la feature
└── tasks.md              # Lista de tareas (documentadas retroactivamente)
```

### Source Code (repository root)

```text
EDA/
├── eda.py                        # Script principal (carga, tablas, gráficas)
├── informe_eda.md                # Informe con conclusiones
├── estadisticas_descriptivas.csv
├── balance_clases.csv
├── conteo_modos_falla.csv
├── conteo_tipo_producto.csv
└── figures/
    ├── balance_clases.png
    ├── modos_falla.png
    ├── distribuciones.png
    ├── boxplots_por_falla.png
    └── correlaciones.png
```

**Structure Decision**: Se usó una carpeta `EDA/` propia en la raíz del repo (en vez de `notebooks/` + `docs/eda/` como se planteó inicialmente) para mantener el script, las tablas, las gráficas y el informe consolidados en un solo lugar, fácil de revisar y aislado del resto de carpetas (`src/`, `app/`, `docs/`) que ya usa el resto del equipo.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| No hay tests en `tests/` para `EDA/eda.py` (Principio V) | Es un script de análisis de un solo uso que genera artefactos para un informe, no una librería con lógica de negocio reutilizable por otros módulos | Escribir tests AAA para funciones de plotting (`plot_*`) aportaría poco valor (verificarían solo que se llama a matplotlib, no la calidad del análisis); se prefiere verificación manual de las tablas/gráficas generadas + `ruff check` para calidad de código |
