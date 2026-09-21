# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Mantenimiento Predictivo IA — Makefile                                    ║
# ║  Ejecutar `make help` para ver todos los comandos disponibles.             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

.DEFAULT_GOAL := help
SHELL := cmd.exe

# ── Variables ──────────────────────────────────────────────────────────────────

PYTHON   := uv run python
PYTEST   := uv run pytest
RUFF     := uv run ruff
STREAMLIT:= uv run streamlit
APP_FILE := app/main.py
PORT     := 8501

# Backend de MLflow: debe coincidir con TRACKING_URI_POR_DEFECTO en
# src/evaluation/tracking.py, o la UI abre un store vacio sin ningun run.
MLFLOW_URI  := sqlite:///mlflow.db
MLFLOW_PORT := 5000

# API key de TabPFN (ux.priorlabs.ai/account) para docker-run. Se toma del
# entorno si ya esta exportada; de lo contrario, `make TABPFN_TOKEN=... docker-run`.
TABPFN_TOKEN ?=

# Servicio gRPC de inferencia (ver proto/mantenimiento.proto). GRPC_PORT debe
# coincidir con el puerto que escucha src/serving/server.py.
GRPC_PORT := 50051

# Stage del Dockerfile a construir/ejecutar con docker-build/docker-run: `web`
# (por defecto) o `inference`. Ej: `make DOCKER_TARGET=inference docker-build`.
DOCKER_TARGET ?= web

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HELP — Menú de comandos                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

.PHONY: help
help: ## Muestra este menú de ayuda
	@echo.
	@echo  Mantenimiento Predictivo IA — Comandos disponibles
	@echo  ────────────────────────────────────────────────────
	@echo.
	@echo  Uso: make [comando]
	@echo.
	@echo  Desarrollo:
	@echo    install            Instala dependencias con uv sync
	@echo    lint               Verifica estilo con Ruff
	@echo    lint-fix           Corrige problemas de estilo automaticamente
	@echo    format             Formatea el codigo con Ruff
	@echo    format-check       Verifica formato sin modificar (CI)
	@echo    check              lint + format-check (para CI)
	@echo    test               Ejecuta todos los tests
	@echo    test-unit          Solo tests unitarios
	@echo    test-integration   Solo tests de integracion
	@echo    test-app           Tests de la interfaz Streamlit
	@echo    coverage           Tests con reporte de cobertura
	@echo.
	@echo  Aplicacion (servicios gRPC, ver proto/mantenimiento.proto):
	@echo    grpc-server        Lanza el servicio de inferencia TabPFN-v2 (puerto 50051)
	@echo    app                Lanza la interfaz web (Streamlit, cliente del servicio anterior)
	@echo    app-reload         Lanza la interfaz con hot-reload forzado
	@echo    proto              Regenera los stubs gRPC desde proto/mantenimiento.proto
	@echo    evaluar            Evalua TabPFN + XGBoost (MLflow)
	@echo    evaluar-xgboost    Evalua solo XGBoost
	@echo.
	@echo  Docker:
	@echo    docker-build       Construye un stage suelto (DOCKER_TARGET=web^|inference)
	@echo    docker-run         Ejecuta ese stage suelto en un contenedor
	@echo    compose-up         Levanta web + inference juntos con docker compose
	@echo    compose-down       Detiene los servicios de docker compose
	@echo.
	@echo  Utilidades:
	@echo    clean              Limpia caches y archivos temporales
	@echo    pre-commit         Ejecuta pre-commit en todos los archivos
	@echo    mlflow             Abre la UI de MLflow
	@echo    status             Estado del proyecto (Python, git, deps)
	@echo.

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DESARROLLO — Linting, formato y tests                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

.PHONY: install
install: ## Instala todas las dependencias con uv sync
	uv sync

.PHONY: lint
lint: ## Verifica estilo con Ruff (sin modificar archivos)
	$(RUFF) check .

.PHONY: lint-fix
lint-fix: ## Corrige automáticamente problemas de estilo con Ruff
	$(RUFF) check --fix .

.PHONY: format
format: ## Formatea el código con Ruff
	$(RUFF) format .

.PHONY: format-check
format-check: ## Verifica formato sin modificar (para CI)
	$(RUFF) format --check .

.PHONY: check
check: lint format-check ## Ejecuta lint + format-check (ideal para CI)

.PHONY: test
test: ## Ejecuta todos los tests con pytest
	$(PYTEST) -v

.PHONY: test-unit
test-unit: ## Ejecuta solo tests unitarios
	$(PYTEST) tests/unit/ -v

.PHONY: test-integration
test-integration: ## Ejecuta solo tests de integración
	$(PYTEST) tests/integration/ -v

.PHONY: test-app
test-app: ## Ejecuta tests de la interfaz Streamlit
	$(PYTEST) tests/unit/test_app.py -v

.PHONY: coverage
coverage: ## Ejecuta tests con reporte de cobertura
	$(PYTEST) --cov=src --cov=app --cov-report=term-missing -v

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  APLICACIÓN — Servicios gRPC, interfaz web y evaluación de modelos         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

.PHONY: proto
proto: ## Regenera los stubs gRPC desde proto/mantenimiento.proto
	uv run python -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto --pyi_out=proto proto/mantenimiento.proto
	@$(PYTHON) -c "import pathlib; p = pathlib.Path('proto/mantenimiento_pb2_grpc.py'); s = p.read_text(); p.write_text(s.replace('import mantenimiento_pb2 as mantenimiento__pb2', 'from proto import mantenimiento_pb2 as mantenimiento__pb2'))"

.PHONY: grpc-server
grpc-server: ## Lanza el servicio de inferencia gRPC (TabPFN-v2, puerto 50051)
	$(PYTHON) -m src.serving.server

.PHONY: app
app: ## Lanza la interfaz web de Streamlit (requiere `grpc-server` corriendo aparte)
	$(STREAMLIT) run $(APP_FILE) --server.port=$(PORT)

.PHONY: app-reload
app-reload: ## Lanza la interfaz con hot-reload forzado
	$(STREAMLIT) run $(APP_FILE) --server.port=$(PORT) --server.runOnSave=true

.PHONY: evaluar
evaluar: ## Evalúa TabPFN-v2 y XGBoost, registra en MLflow
	$(PYTHON) -m scripts.evaluar_modelos

.PHONY: evaluar-xgboost
evaluar-xgboost: ## Evalúa solo XGBoost (sin TabPFN)
	$(PYTHON) -m scripts.evaluar_modelos --sin-tabpfn

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DOCKER — Contenedores                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

.PHONY: docker-build
docker-build: ## Construye un stage suelto del Dockerfile (ver DOCKER_TARGET)
	docker build --target $(DOCKER_TARGET) -t mantenimiento-predictivo-$(DOCKER_TARGET) .

.PHONY: docker-run
docker-run: ## Ejecuta ese stage suelto (monta ./data y reenvia TABPFN_TOKEN; solo compose conecta los dos servicios)
	docker run --rm -p $(PORT):8501 -p $(GRPC_PORT):$(GRPC_PORT) -v "$(CURDIR)/data:/app/data" -e TABPFN_TOKEN=$(TABPFN_TOKEN) -e GRPC_PORT=$(GRPC_PORT) mantenimiento-predictivo-$(DOCKER_TARGET)

.PHONY: compose-up
compose-up: ## Levanta web + inference juntos con docker compose (comunicados por gRPC)
	docker compose up --build

.PHONY: compose-down
compose-down: ## Detiene y elimina los contenedores de docker compose
	docker compose down

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  UTILIDADES — Limpieza y mantenimiento                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

.PHONY: clean
clean: ## Limpia archivos temporales, cachés y __pycache__
	@echo Limpiando archivos temporales...
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist .ruff_cache rmdir /s /q .ruff_cache
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@echo Limpieza completada.

.PHONY: pre-commit
pre-commit: ## Ejecuta pre-commit en todos los archivos
	uv run pre-commit run --all-files

.PHONY: mlflow
mlflow: ## Abre la UI de MLflow en el navegador
	uv run mlflow ui --backend-store-uri $(MLFLOW_URI) --port $(MLFLOW_PORT)

.PHONY: status
status: ## Muestra el estado del proyecto (Python, dependencias, git)
	@echo ── Python ──
	@$(PYTHON) --version
	@echo.
	@echo ── Git ──
	@git branch --show-current
	@git status --short
	@echo.
	@echo ── Dependencias ──
	@uv pip list --quiet 2>nul | findstr /i "streamlit tabpfn xgboost scikit"
