# AGENTS.md — Reglas de revisión de código

Reglas que usa Gentleman Guardian Angel (GGA) para revisar cada commit de código Python. Son un resumen orientado a revisión; ante cualquier duda, la fuente de verdad es la constitución del proyecto en `.specify/memory/constitution.md`.

## Estilo y calidad

- PEP 8 y cero warnings, verificado con Ruff (`uv run ruff check .` / `uv run ruff format .`).
- Toda función, método y clase pública debe tener docstring en formato Google (resumen en imperativo + `Args:` / `Returns:` / `Raises:` según aplique). Ruff lo exige automáticamente (`convention = "google"`).
- Funciones cortas, una sola responsabilidad, alta cohesión y bajo acoplamiento.
- Type hints en las firmas de funciones, especialmente en `src/`.

## Pruebas

- Todo test de pytest sigue la estructura AAA, con los comentarios literales `# 1. ARRANGE`, `# 2. ACT`, `# 3. ASSERT`.
- Los tests no necesitan `Args:`/`Returns:` en su docstring — quedan documentados por la propia estructura AAA.
- Todo módulo en `src/` debe tener sus pruebas correspondientes en `tests/`.

## Modelado y métricas

- Nunca usar accuracy como métrica de éxito (el dataset tiene ~3.4% de clase "falla"). Usar F1, Recall y PR-AUC sobre la clase falla.
- TabPFN-v2 es el modelo base: no se entrena por gradiente ni se ajustan hiperparámetros vía GridSearch/RandomSearch.

## Dependencias y estructura

- Todas las dependencias se agregan con `uv add` / `uv add --dev`, nunca con `pip install` directo.
- No commitear binarios pesados, checkpoints de modelo, ni archivos `.ipynb` con salidas grandes.
- Cada incremento de trabajo vive en su propia carpeta `specs/<NNN>-<nombre>/` (spec-kit), según define la constitución.

## Commits y ramas

- Mensajes de commit en formato `tipo: descripción breve` (`feat`, `fix`, `chore`, `docs`, `test`, `refactor`).
- Trabajo en ramas propias (por feature o por integrante), nunca directo a `main`.

## Revisión de IA

- Todo código generado o asistido por IA debe poder ser explicado por su autor en sustentación; no copiar código de otros equipos o repos sin entenderlo.
