# Mantenimiento Predictivo con TabPFN-v2

![Estado](https://img.shields.io/badge/estado-en%20desarrollo-yellow)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Gestor](https://img.shields.io/badge/dependencias-uv-purple)
![Licencia](https://img.shields.io/badge/uso-acad%C3%A9mico-lightgrey)

Sistema de **mantenimiento predictivo** para maquinaria industrial que estima la
probabilidad de fallo de un equipo a partir de sus variables operativas (temperatura,
velocidad de rotación, torque, desgaste de herramienta, etc.).

El modelo base es **TabPFN-v2**, un transformer preentrenado para clasificación
tabular que no requiere entrenamiento tradicional. Como *baselines* opcionales se
contemplan **XGBoost**, **Gradient Boosting** y **TabPFN-Mix**. La solución se
expone mediante una interfaz web en **Streamlit** y se empaqueta con **Docker**.

El sistema está dividido en dos servicios que se comunican por **gRPC**: `inference`
carga TabPFN-v2 (y su stack pesado: torch, tabpfn) y expone la predicción por la red;
`web` es la interfaz Streamlit, un cliente gRPC ligero de ese servicio. Ver
`proto/mantenimiento.proto` para el contrato y `docker-compose.yml` para levantar
ambos juntos.

---

## Integrantes

| Integrante                |
|---------------------------|
| Juliana Campuzano         |
| Diego Fernando Garcés     |
| Jorge Iván Gonzáles       |
| Juan José Mosquera        |

**Universidad:** Universidad Autónoma de Occidente
**Curso:** Desarrollo de Proyectos en IA — Módulo 2

---

## Tecnologías

| Componente            | Herramienta                                   |
|-----------------------|-----------------------------------------------|
| Modelo base           | TabPFN-v2                                      |
| Baselines opcionales  | XGBoost · Gradient Boosting · TabPFN-Mix       |
| Dataset               | AI4I 2020 Predictive Maintenance Dataset (UCI) |
| Procesamiento         | pandas · NumPy · scikit-learn                  |
| Visualización         | matplotlib                                     |
| Interfaz              | Streamlit (cliente gRPC)                       |
| Comunicación entre servicios | gRPC (Protocol Buffers)                 |
| Contenedor            | Docker (`python:3.12-slim` + `uv`) + Docker Compose |
| Gestión de entorno    | uv                                             |
| Pruebas               | pytest                                         |
| Seguimiento de experimentos | MLflow                                   |

---

## Estructura del proyecto

```
.
├── data/
│   ├── raw/            # Datos originales sin procesar (AI4I 2020)
│   └── processed/      # Datos limpios y transformados
├── notebooks/          # Exploración y prototipado
├── EDA/                # Análisis exploratorio: script, tablas, figuras e informe
├── src/
│   ├── preprocessing/  # Limpieza, ingeniería de características, splits
│   ├── models/         # TabPFN-v2 y baselines
│   ├── evaluation/     # Métricas, curvas, comparativas
│   └── serving/        # Servicio gRPC de inferencia (servidor + carga del modelo)
├── proto/              # Contrato gRPC (.proto) y stubs generados (make proto)
├── app/                # Interfaz Streamlit (cliente gRPC de `inference`)
├── scripts/            # Entrypoints ejecutables (evaluación de modelos)
├── specs/              # Especificaciones spec-kit por incremento
├── docker/             # Recursos de contenedorización
├── tests/              # Pruebas con pytest
├── docs/               # Documentación del proyecto
├── Dockerfile          # Multi-stage: servicios `web` e `inference`
├── docker-compose.yml  # Orquesta web + inference (gRPC entre ambos)
├── pyproject.toml      # Dependencias (uv)
└── README.md
```

---

## Instalación

Requisitos: [uv](https://docs.astral.sh/uv/) y Python 3.12.

```bash
# 1. Clonar el repositorio
git clone https://github.com/julicapa15/ProyectoDDPIA_MantenimientoPredictivo.git
cd ProyectoDDPIA_MantenimientoPredictivo

# 2. Crear el entorno e instalar dependencias
uv sync

# 3. (Opcional) Activar el entorno
#    Windows:  .venv\Scripts\activate
#    Linux/Mac: source .venv/bin/activate
```

Para añadir nuevas dependencias:

```bash
uv add <paquete>
```

---

## Ejecución de la aplicación

La app son dos servicios que se hablan por gRPC (ver `proto/mantenimiento.proto`):
`inference` carga TabPFN-v2, `web` es la interfaz Streamlit. En dos terminales:

```bash
# Terminal 1: servicio de inferencia (gRPC, puerto 50051)
uv run python -m src.serving.server        # o: make grpc-server

# Terminal 2: interfaz web (cliente gRPC del anterior)
uv run streamlit run app/main.py           # o: make app
```

La interfaz queda disponible en `http://localhost:8501` y muestra si logró conectar
con el servicio de inferencia ("🟢 conectado" / "🔴 no disponible"). Sin
`grpc-server` corriendo, la app arranca igual pero la predicción falla con un
mensaje explícito en vez de colgarse.

---

## Ejecución con Docker

Requisitos: [Docker](https://docs.docker.com/get-docker/) y el dataset AI4I 2020 en
`data/raw/ai4i2020.csv` (no se incluye en la imagen `inference`, ver `.dockerignore`:
se monta como volumen igual que en desarrollo local).

```bash
# Copiar y completar el token de TabPFN (ver .env.example)
cp .env.example .env

# Levantar los dos servicios juntos (web + inference, comunicados por gRPC)
docker compose up --build     # o: make compose-up
```

La interfaz queda disponible en `http://localhost:8501`; el servicio de inferencia,
en `localhost:50051`. Sin `TABPFN_TOKEN` (o sin licencia aceptada en
[ux.priorlabs.ai](https://ux.priorlabs.ai)), la carga del modelo en `inference` falla
al iniciar — ver la nota de Windows/TabPFN más abajo en
[Resultados](#resultados-feature-001--tabpfn-v2-vs-baseline-xgboost).

También puede construirse/ejecutarse un servicio suelto con `make docker-build` /
`make docker-run` (usan el stage `web` por defecto; `make DOCKER_TARGET=inference
docker-build` para el otro) — útil para depurar una imagen a la vez, pero solo
`docker compose` conecta ambos servicios entre sí.

---

## Pruebas

```bash
uv run pytest
```

---

## Análisis exploratorio (EDA)

Análisis descriptivo del dataset previo al modelado: balance de clases, modos de falla,
distribuciones, comparación de variables según ocurrencia de falla y correlaciones.

```bash
uv run python EDA/eda.py
```

Genera cuatro tablas en `EDA/` y cinco figuras en `EDA/figures/`. El informe completo
con las conclusiones está en **[EDA/informe_eda.md](EDA/informe_eda.md)**.

Hallazgos que condicionan el modelado:

- **Desbalance severo:** 339 fallos sobre 10.000 registros (3,39%), lo que descarta
  *accuracy* como métrica y justifica F1, Recall y PR-AUC sobre la clase falla.
- **Modos de falla desiguales:** HDF (115), OSF (98) y PWF (95) concentran los casos;
  TWF (46) y RNF (19) son marginales.
- **Torque y desgaste son las variables que más discriminan:** en los registros con
  falla el torque medio sube de 39,6 a 50,2 Nm y el desgaste de 106,7 a 143,8 min,
  mientras la velocidad baja de 1540 a 1496 rpm.
- **Dos pares casi colineales:** temperatura de aire y de proceso (r = 0,88) y
  velocidad de rotación con torque (r = −0,88). TabPFN-v2 lo maneja internamente, sin
  selección manual de variables.
- **RNF es ruido por definición:** no debería ser predecible desde las variables de
  proceso; conviene tenerlo presente al interpretar el desempeño por modo de falla.

---

## Resultados: Feature 001 — TabPFN-v2 vs. baseline XGBoost

Clasificación binaria de `Machine failure` sobre AI4I 2020 (10.000 registros, ~3,4% de
fallos). Split estratificado 80/20 con semilla fija; métricas medidas sobre los 2.000
registros de prueba (68 fallos reales), siempre sobre la clase falla —nunca *accuracy*,
que con este desbalance premiaría a un modelo que jamás alerta.

| Modelo        | F1        | Recall    | PR-AUC    |
|---------------|-----------|-----------|-----------|
| **TabPFN-v2** | **0,825** | **0,765** | **0,880** |
| XGBoost       | 0,729     | 0,750     | 0,837     |

**TabPFN-v2 supera al baseline en las tres métricas**, sin entrenamiento por gradiente y
sin ajuste de hiperparámetros: solo aprendizaje en contexto. XGBoost compite de cerca en
recall (0,750 vs. 0,765) tras compensar el desbalance con `scale_pos_weight` = 28,52.

Reproducir la comparativa y registrar ambos runs en MLflow:

```bash
uv run python -m scripts.evaluar_modelos
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

> TabPFN-v2 descarga sus pesos tras una aceptación de licencia única en
> [ux.priorlabs.ai](https://ux.priorlabs.ai). Sin ella, sus pruebas se omiten
> automáticamente. Detalle en
> [specs/001-tabpfn-xgboost-baseline/quickstart.md](specs/001-tabpfn-xgboost-baseline/quickstart.md).

> **En Windows**, ese flujo interactivo falla con `OSError: [WinError 10038]`: la
> librería usa `select()` sobre `stdin`, que en Windows solo admite sockets. La
> solución es aceptar la licencia y generar una API key en
> [ux.priorlabs.ai/account](https://ux.priorlabs.ai/account), y exportarla antes de
> ejecutar — en PowerShell, `$env:TABPFN_TOKEN = "tu_api_key"`, con comillas.

---

## Estado del proyecto

🚧 **En desarrollo — Módulo 2.**

| Fase                                      | Estado        |
|-------------------------------------------|---------------|
| Estructura del repositorio                | ✅ Completado |
| Carga y exploración del dataset AI4I 2020 | ✅ Completado |
| Preprocesamiento y *feature engineering*  | ✅ Completado |
| Modelo base TabPFN-v2                     | ✅ Completado |
| Baseline XGBoost                          | ✅ Completado |
| Baselines opcionales (GB / TabPFN-Mix)    | ⏳ Pendiente  |
| Evaluación y comparativa                  | ✅ Completado |
| Interfaz Streamlit                        | ⏳ Pendiente  |
| Contenedorización con Docker              | ✅ Completado |
