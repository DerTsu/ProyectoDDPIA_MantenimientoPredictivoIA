# syntax=docker/dockerfile:1
#
# Dockerfile — Mantenimiento Predictivo IA
#
# Multi-stage con dos imágenes finales — los dos servicios que se comunican
# por gRPC (ver proto/mantenimiento.proto):
#   - `inference`: carga TabPFN-v2 (torch/tabpfn) y expone PredictFailure/
#     HealthCheck en el puerto 50051.
#   - `web`: interfaz Streamlit, cliente gRPC ligero del servicio anterior,
#     puerto 8501.
# Comparten la etapa `base` (dependencias + código) para no reinstalar nada
# dos veces; solo difieren en EXPOSE/HEALTHCHECK/CMD.
#
# Build:
#   docker build --target inference -t mantenimiento-predictivo-inference .
#   docker build --target web       -t mantenimiento-predictivo-web .
# (o, más simple, `docker compose up --build` — ver docker-compose.yml)
#
# El dataset AI4I 2020 (data/raw/ai4i2020.csv) está excluido de la imagen
# `inference` (ver .dockerignore, igual que en .gitignore) y debe montarse
# como volumen en tiempo de ejecución — ver README, sección "Ejecución con
# Docker".

FROM python:3.12-slim AS base

# Binario estático de uv, copiado directo desde su imagen oficial (sin pip).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# 1) Capa de dependencias: se cachea mientras pyproject.toml/uv.lock no cambien,
#    sin importar cambios posteriores en el código de app/, src/ o proto/.
#    `--no-dev` excluye pytest/ruff/pre-commit/grpcio-tools, innecesarios en
#    producción (los stubs de proto/ ya vienen generados y committeados).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2) Código del proyecto necesario en tiempo de ejecución (tests/, notebooks/,
#    EDA/ y los datos crudos quedan fuera vía .dockerignore).
COPY proto/ ./proto/
COPY app/ ./app/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Usuario sin privilegios. TabPFN-v2 descarga sus pesos preentrenados en el
# primer uso bajo $HOME/.cache, por eso HOME apunta a un directorio de un
# usuario real (no root) con permisos de escritura. Se crean además los
# subdirectorios de datos para que el volumen montado en /app/data (solo lo
# usa `inference`) tenga dueño correcto.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --home-dir /home/app --create-home app \
    && mkdir -p /app/data/raw /app/data/processed \
    && chown -R app:app /app /home/app
ENV HOME=/home/app
USER app

# ── inference: servicio gRPC (TabPFN-v2), puerto 50051 ──────────────────────
FROM base AS inference

EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-m", "src.serving.healthcheck"]

CMD ["python", "-m", "src.serving.server"]

# ── web: interfaz Streamlit ligera (sin PyTorch ni TabPFN) ───────────────────
FROM python:3.12-slim AS web

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Instala únicamente las librerías necesarias para la web (~350 MB)
RUN uv pip install --system streamlit grpcio numpy pandas scikit-learn

COPY proto/ ./proto/
COPY app/ ./app/
COPY src/ ./src/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]