# Plan de Implementación: Interfaz Web Streamlit Modular con TabPFN-v2

**Rama**: `docs/spec-003-interfaz-streamlit` | **Fecha**: 2026-09-18 | **Spec**: `specs/003-interfaz-streamlit/spec.md`

**Entrada**: Modularizar `app/main.py` desacoplando la lógica de dominio y los componentes de presentación para cumplir los principios de alta cohesión, bajo acoplamiento y testeabilidad de la constitución.

## Resumen

Evolucionar la interfaz de Streamlit dividiendo el archivo monolítico `app/main.py` en módulos especializados:
1. `app/utils/physics.py`: Funciones puras de ingeniería física ($\Delta T$, potencia mecánica $\tau \cdot \omega$, etc.).
2. `app/utils/preprocessing.py`: Funciones puras de preparación de features para el modelo (one-hot y validación de rangos).
3. `app/utils/model.py`: Gestión de caché con `@st.cache_resource`, carga de contexto y llamadas al estimador TabPFN-v2.
4. `app/components/`: Componentes reutilizables de UI (formulario de entrada y visualización de resultados).
5. `app/main.py`: Orquestador principal ligero.

## Contexto Técnico

- **Lenguaje/Versión**: Python 3.12 (uv)
- **Dependencias**: `streamlit>=1.63.0`, `tabpfn>=8.5.0`, `scikit-learn>=1.9.0`, `numpy>=2.4.6`, `pandas>=3.0.5`
- **Testing**: `tests/unit/test_app.py` ejecutado con `pytest` bajo estructura AAA.
- **Calidad**: PEP 8 verificado con `ruff check .` y `ruff format --check .`.
- **Plataforma Objetivo**: CPU local (contexto estratificado a 1.000 muestras para latencia <= 2 s).

## Arquitectura Modular Propuesta

```text
app/
├── __init__.py
├── main.py                  # Orquestador y layout principal
├── components/              # Elementos visuales de Streamlit
│   ├── __init__.py
│   ├── form.py              # Formulario interactivo st.form
│   └── results.py           # Renderizado de alertas, métricas y detalles técnicos
└── utils/                   # Lógica de dominio y funciones puras
    ├── __init__.py
    ├── physics.py           # Cálculo de potencia y Delta T
    ├── preprocessing.py     # Preparación de vectores (1, 8) y niveles de alerta
    └── model.py             # Carga y fit en caché de TabPFN-v2
```
