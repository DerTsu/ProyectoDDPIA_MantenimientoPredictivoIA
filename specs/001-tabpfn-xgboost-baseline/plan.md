# Plan de Implementación: TabPFN-v2 + Baseline XGBoost

**Rama**: `001-tabpfn-xgboost-baseline` | **Fecha**: 2026-09-13 | **Spec**: `specs/001-tabpfn-xgboost-baseline/spec.md`

**Entrada**: Especificación de feature desde `/specs/001-tabpfn-xgboost-baseline/spec.md`

## Resumen

Integrar TabPFN-v2 (preentrenado, in-context) como modelo base para clasificar fallos de máquinas en dataset AI4I 2020 (10k registros, 6 variables de proceso, ~3% tasa de falla). Entrenar XGBoost como baseline clásico para comparación objetiva. Evaluar ambos con F1, Recall, PR-AUC sobre la clase minoría (falla). Loguear ambos runs a MLflow con reproducibilidad total.

## Contexto Técnico

**Lenguaje/Versión**: Python 3.12 (uv)

**Dependencias Principales**:
- `tabpfn>=8.5.0` (preentrenado, basado en PyTorch)
- `xgboost>=3.2.0` (compatible con sklearn)
- `scikit-learn>=1.9.0` (métricas, preprocesamiento, split train/test)
- `pandas>=3.0.5` (carga de datos)
- `numpy>=2.4.6` (arrays)
- `mlflow>=3.16.0` (tracking de experimentos)
- `matplotlib>=3.11.1` (visualización, opcional)

**Almacenamiento**: Archivos locales
- Entrada: `data/raw/ai4i2020.csv` (10,000 filas × 7 columnas)
- Salida: Modelos serializados (`.safetensors` para TabPFN, `.json` para XGBoost) + artefactos de runs en MLflow

**Testing**: pytest (ya en el proyecto)

**Plataforma Objetivo**: CPU local (sin GPU requerida)

**Tipo de Proyecto**: Pipeline de clasificación ML + estudio comparativo

**Objetivos de Rendimiento**:
- Inferencia TabPFN: <5 segundos para 2000 muestras de test (CPU)
- Entrenamiento XGBoost: <10 segundos (CPU)
- Evaluación completa: <30 segundos total

**Restricciones**:
- Sin ajuste de hiperparámetros para TabPFN (preentrenado, solo in-context)
- XGBoost class_weight debe manejar ~3% de clase falla

**Escala/Alcance**:
- Dataset: 10k registros, 6 features (tras one-hot: 7-8 columnas)
- Modelos: 1 preentrenado (TabPFN-v2) + 1 entrenado desde cero (XGBoost)
- Métricas: 3 por modelo (F1, Recall, PR-AUC) = 6 métricas totales

## Verificación de Constitución

✅ **Principio I (TabPFN-v2 como base, sin ajuste por gradiente)**: PASA. Solo aprendizaje in-context, sin GridSearch.
✅ **Principio II (F1, Recall, PR-AUC solo)**: PASA. Explícitamente computadas, sin accuracy.
✅ **Principio III (Preprocesamiento mínimo)**: PASA. One-hot Type, split estratificado, StandardScaler solo para XGBoost si aplica.
✅ **Principio IV (CRISP-DM)**: PASA. Prep datos → Modelado → Evaluación → logging MLflow.
✅ **Principio V (Tests con estructura AAA)**: PASA. Los 53 tests de `tests/` usan los bloques `# 1. ARRANGE`, `# 2. ACT` y `# 3. ASSERT`.
✅ **Matriz de Alcance**: PASA. Incluido: TabPFN-v2, XGBoost hoy, F1/recall/PR-AUC. Excluido (opcional): GradientBoosting, TabPFN-Mix.
✅ **Stack**: Python 3.12, uv, sklearn/xgboost/tabpfn, mlflow. Todas las dependencias declaradas en pyproject.toml.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
# [REMOVE IF UNUSED] Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# [REMOVE IF UNUSED] Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# [REMOVE IF UNUSED] Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure: feature modules, UI flows, platform tests]
```

**Structure Decision**: [Document the selected structure and reference the real
directories captured above]

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
