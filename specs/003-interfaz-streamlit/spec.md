# Feature Specification: Interfaz Web Streamlit con TabPFN-v2

**Feature Branch**: `docs/spec-003-interfaz-streamlit` (registrada como `specs/003-interfaz-streamlit/`)

**Created**: 2026-09-18

**Status**: Especificado / Implementado

**Input**: Desarrollar una interfaz gráfica interactiva en Streamlit para el sistema de mantenimiento predictivo industrial sobre el dataset AI4I 2020, consumiendo TabPFN-v2 como clasificador in-context en tiempo real, optimizado para CPU y desacoplado en componentes modulares.

## User Scenarios & Testing

### User Story 1 - Operador industrial realiza diagnóstico de telemetría (Priority: P1)

Un operador de planta ingresa en la aplicación web las 6 variables operacionales de una máquina (tipo de producto, temperatura de aire, temperatura de proceso, RPM, torque y desgaste de herramienta) y obtiene en menos de 3 segundos la probabilidad de falla y una alerta con semáforo visual (Normal, Precaución, Falla inminente), además de métricas físicas derivadas ($\Delta T$ y potencia).

**Why this priority**: Es la funcionalidad central de usuario final para monitoreo predictivo.

**Independent Test**:
1. Se ingresan valores nominales seguros -> Alerta "Normal" (verde).
2. Se ingresan valores anómalos de torque (>70 Nm) o desgaste (>200 min) -> Alerta "Precaución" / "Falla inminente".

**Acceptance Scenarios**:
1. **Given** la aplicación en ejecución (`uv run streamlit run app/main.py`), **When** se cargan los recursos iniciales, **Then** el modelo TabPFN-v2 y el contexto se almacenan en caché (`@st.cache_resource`) sin bloquear la UI.
2. **Given** el formulario de entrada, **When** el usuario presiona "Predecir falla", **Then** la inferencia se completa en menos de 3 segundos en CPU gracias al contexto estratificado.
3. **Given** la inferencia completada, **When** se despliega el resultado, **Then** se muestra la probabilidad porcentual, el nivel de alerta, Delta T y potencia mecánica en watts.

---

### User Story 2 - Desarrollador mantiene y testea la interfaz de forma modular (Priority: P1)

El equipo de desarrollo necesita que la aplicación esté modularizada en componentes y utilidades reutilizables (`app/components/`, `app/utils/`) con funciones puras testables aisladas del runtime de Streamlit.

**Why this priority**: Cumplir con la constitución y AGENTS.md (alta cohesión, bajo acoplamiento, cero warnings y pruebas unitarias completas bajo patrón AAA).

**Acceptance Scenarios**:
1. **Given** los módulos de cálculo y preprocesamiento, **When** se ejecutan las pruebas con `uv run pytest tests/unit/test_app.py`, **Then** todas las pruebas pasan sin requerir inicializar el servidor Streamlit.
2. **Given** la suite de linters, **When** se ejecuta `uv run ruff check .` y `uv run ruff format --check .`, **Then** se reportan 0 errores de estilo y formato.

---

## Requirements

### Functional Requirements

- **FR-001**: La interfaz DEBE permitir el ingreso interactivo de las 6 variables operacionales del dataset AI4I 2020.
- **FR-002**: El sistema DEBE preprocesar los datos al vector de 8 características requeridas por el modelo (`Type_H`, `Type_L`, `Type_M`, y las 5 variables continuas).
- **FR-003**: El sistema DEBE emplear TabPFN-v2 sin ajuste por gradiente, cargando un contexto estratificado en memoria optimizado para CPU.
- **FR-004**: El sistema DEBE clasificar el riesgo en tres categorías mediante umbrales parametrizados: Normal (< 30%), Precaución (30% - 70%) y Falla inminente (>= 70%).
- **FR-005**: El sistema DEBE calcular dinámicamente Delta T y potencia mecánica.
- **FR-006**: La arquitectura DEBE estructurarse modularmente dividiendo `main.py` en submódulos dedicados para componentes visuales, carga de modelos y funciones de dominio.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: Inferencia interactiva en CPU completada en <= 3.0 segundos.
- **SC-002**: Cobertura de tests unitarios al 100% de las funciones puras en `tests/unit/test_app.py`.
- **SC-003**: Cero warnings en `uv run ruff check .` y cero diferencias en `uv run ruff format --check .`.
