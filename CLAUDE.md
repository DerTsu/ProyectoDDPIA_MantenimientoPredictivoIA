# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Predictive maintenance system for industrial machinery (AI4I 2020 dataset) that estimates
failure probability from operating variables (temperature, rotational speed, torque, tool
wear). Base model is **TabPFN-v2** (pretrained tabular transformer, in-context learning, no
gradient training); **XGBoost** is the classical baseline. Split into two services that
talk over **gRPC**: `inference` (loads TabPFN-v2) and `web` (a **Streamlit** UI, a thin
gRPC client). Academic project for Universidad Autónoma de Occidente, built with spec-kit
(spec-driven development) across incremental features under `specs/<NNN>-<nombre>/`.

## Commands

Dependency manager is `uv`; always run Python through `uv run`, never a bare `python`/`pip`.
Add dependencies with `uv add` / `uv add --dev` — never `pip install` directly.

```bash
uv sync                                   # install dependencies
uv run python -m src.serving.server       # run the inference gRPC service (or: make grpc-server)
uv run streamlit run app/main.py          # run the web UI (or: make app) — needs the above running
uv run pytest                             # run all tests
uv run pytest tests/unit/test_xgboost.py  # run a single test file
uv run pytest tests/unit/test_xgboost.py::test_name -v   # run a single test
uv run pytest tests/unit/ -v              # unit tests only
uv run pytest tests/integration/ -v       # integration tests only
uv run pytest --cov=src --cov=app --cov-report=term-missing -v  # coverage
uv run ruff check .                       # lint
uv run ruff check --fix .                 # lint with autofix
uv run ruff format .                      # format
uv run python -m scripts.evaluar_modelos             # evaluate TabPFN + XGBoost, log to MLflow
uv run python -m scripts.evaluar_modelos --sin-tabpfn  # evaluate XGBoost only
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000  # MLflow UI
uv run python EDA/eda.py                  # regenerate EDA tables/figures
uv run python -m grpc_tools.protoc -I proto --python_out=proto \
  --grpc_python_out=proto --pyi_out=proto proto/mantenimiento.proto  # regen gRPC stubs (or: make proto)
docker compose up --build                 # run both services in containers (or: make compose-up)
```

A `Makefile` wraps most of these (`make help` lists all targets, e.g. `make test`,
`make lint`, `make app`, `make grpc-server`, `make proto`, `make compose-up`,
`make evaluar`, `make coverage`, `make status`).

TabPFN-v2 needs a one-time license acceptance at ux.priorlabs.ai; without it, TabPFN
tests/fixtures skip automatically (`TabPFNLicenseError` is caught in `tests/conftest.py`).
**On Windows**, the interactive license flow fails with `OSError: [WinError 10038]`
(`select()` on stdin only supports sockets on Windows) — instead generate an API key at
ux.priorlabs.ai/account and set `$env:TABPFN_TOKEN` before running.

## Architecture

**Two services talk over gRPC** (contract in `proto/mantenimiento.proto`, stubs generated
into `proto/*_pb2*.py` by `make proto` — never hand-edited): `inference` (`src/serving/`)
loads TabPFN-v2 and answers `PredictFailure`/`HealthCheck`; `web` (`app/`) is the Streamlit
UI, a thin gRPC client with no TabPFN/torch dependency at runtime. The service boundary
sits exactly where the old in-process call used to be: the client still builds the
preprocessed `(1, 8)` feature vector itself (`app/utils/preprocessing.preprocess_input`)
and only sends that vector over the wire — `inference` does not preprocess, only predicts.

**`src/` is the model/data core, framework-agnostic; `app/` is the Streamlit presentation
layer, `src/serving/` is the gRPC presentation layer for the same core.** Data flows one
direction: `src/preprocessing` → `src/models` → `src/evaluation`; both `app/` and
`src/serving/model_loader.py` build a TabPFN context from `src.preprocessing.build_features`
— `app/` no longer does, now that `src.serving` owns model loading (see below). The
Streamlit app still re-implements a parallel, simplified preprocessing path in `app/utils/`
instead of reusing `src/preprocessing` directly. The feature *order* is not hand-duplicated
— `app/utils/preprocessing.NOMBRE_FEATURES` derives from
`src.preprocessing.preprocess.COLUMNAS_DUMMY + COLUMNAS_NUMERICAS`, and
`tests/unit/test_app.py::test_preprocess_input_produce_los_mismos_valores_que_preprocess_features_de_src`
runs a concrete row through both code paths independently and fails if their 8 output
values (order or content) ever diverge. The one-hot/dict-building *logic* itself
is still duplicated by hand (no pandas dependency in `app/`, unlike `src/`'s `pd.get_dummies`
approach) — that part remains acceptable duplication, not a drift risk, since order is now
pinned to a single source of truth.

- **`src/preprocessing/`**: `load.py` validates and loads the raw AI4I 2020 CSV (exact
  column set, exact row count, no nulls — raises otherwise). `preprocess.py` does the only
  sanctioned transformation: one-hot `Type` → `Type_H/Type_L/Type_M`, drop out-of-scope
  columns (IDs and the 5 failure-submode columns), keep the 5 numeric process variables and
  the `Machine failure` target. `features.py`'s `build_features()` chains both and produces
  the stratified 80/20 train/test split (fixed `random_state=42`) — this is the single
  entrypoint tests and scripts use to go from raw CSV to model-ready arrays.
- **`src/models/`**: both models expose `predict_proba(...) -> (n, 2)` and a `save()` that
  serializes in the model's own native format (not pickle). `TabPFNClassifier` is a thin
  wrapper: `predict_proba(X_train, y_train, X_test)` takes the training set as part of the
  call signature (not `fit` then `predict`) because TabPFN has no persistent gradient-fitted
  state — the reference set *is* the context. It also force-enables
  `ignore_pretraining_limits=True` since CPU inference is blocked above 5000 rows by default
  and this project's `X_train` has 8000. `XGBoostBaseline` is conventional
  fit/predict_proba; `from_class_balance(y_train)` derives `scale_pos_weight` from the real
  class imbalance instead of a hardcoded value.
- **`src/evaluation/`**: `metrics.py` computes only F1, Recall, PR-AUC on the failure class
  (see Conventions — accuracy is banned). `tracking.py` is the MLflow integration:
  `log_run_to_mlflow()` tags every run with `spec_id` (which `specs/<NNN>-.../` feature it
  belongs to), model license/repo, dataset hash, environment, and logs example predictions +
  a text evaluation report + the packed model as artifacts. Tracking backend is
  `sqlite:///mlflow.db` (MLflow 3.16+ deprecated the file store), overridable via
  `MLFLOW_TRACKING_URI`.
- **`scripts/evaluar_modelos.py`**: CLI entrypoint that runs the full pipeline
  (load → build_features → fit both models → compute metrics → log to MLflow) end to end;
  this is what CI/reproduction commands and the README's results table come from. Offline,
  unrelated to the gRPC serving path below.
- **`proto/mantenimiento.proto`** + generated `proto/mantenimiento_pb2.py` /
  `_pb2_grpc.py`: the gRPC contract between `web` and `inference`. `InferenceService` has
  two RPCs — `PredictFailure(features: repeated double) -> (proba_no_failure, proba_failure)`
  and `HealthCheck`. Regenerate with `make proto` after editing the `.proto`; the generated
  `_pb2_grpc.py` needs its `import mantenimiento_pb2` patched to
  `from proto import mantenimiento_pb2` (protoc's well-known relative-import gap for
  packaged stubs) — `make proto` does this patch automatically, don't hand-edit it away.
  Both generated files are excluded from Ruff (`extend-exclude` in `pyproject.toml`) since
  they're never hand-edited.
- **`src/serving/`**: the `inference` service. `model_loader.py` is the
  Streamlit-free equivalent of the old `app/utils/model.py` — loads the dataset via
  `src.preprocessing`, fits a 1000-row stratified TabPFN-v2 context once (no
  `st.cache_resource`, since this process has no Streamlit runtime and only needs to load
  once at startup). `server.py`'s `InferenceServicer` implements the two RPCs and `serve()`
  starts the gRPC server (`GRPC_PORT` env var, default 50051). `healthcheck.py` is a tiny
  gRPC client used as the Docker `HEALTHCHECK` for the `inference` image — it calls the
  service's own `HealthCheck` RPC rather than just probing the TCP port, so a container
  where the model failed to load is correctly reported unhealthy.
- **`app/`**: `main.py` is the Streamlit orchestrator/entrypoint. Because Streamlit executes
  this file directly (not as part of the `app` package), it manually inserts the project
  root into `sys.path` *before* importing `app.*` / `src.*` — this is why `app/**` is
  exempted from Ruff's `E402` (imports-not-at-top) in `pyproject.toml`. `app/utils/model.py`
  is now a gRPC **client**, not a model loader: `get_inference_stub()` opens (and
  `@st.cache_resource`-caches) a channel to `INFERENCE_SERVICE_ADDR` (env var, default
  `localhost:50051`; gRPC connects lazily, so caching the stub doesn't require the service
  to be up yet), `predict_failure(X_new)` calls `PredictFailure` and re-shapes the response
  back into the `(1, 2)` array `app/components/results.py` expects, and
  `check_inference_health()` backs the connectivity banner shown on the page. A
  `grpc.RpcError` from either call is translated to `InferenceServiceUnavailableError` (a
  domain exception `main.py` catches to show a friendly `st.error`, instead of leaking gRPC
  specifics into the presentation layer). `app/utils/preprocessing.py` rebuilds the one-hot
  + feature vector for a single form submission by hand (no pandas dependency, just numpy),
  but imports `NOMBRE_FEATURES` from `src.preprocessing.preprocess`'s constants instead of
  hardcoding the column order — see Architecture intro above for the drift test that guards
  this. `app/utils/physics.py` has pure derived-quantity functions (mechanical power, ΔT)
  and the alert-level thresholds (0.30 / 0.70) shown in the UI. `app/components/form.py` and
  `results.py` render the input form and prediction output.

## Conventions (enforced, not optional)

- **Never use accuracy** as a success metric anywhere (dataset is ~3.4% failure class) — only
  F1, Recall, PR-AUC on the failure class. This is a project constitution principle
  (`.specify/memory/constitution.md`), not a style preference.
- **TabPFN-v2 is never retrained or hyperparameter-tuned** (no GridSearch/RandomSearch, no
  gradient fitting) — it's used purely in-context. If asked to "improve" the TabPFN path,
  the answer is not training it further.
- Every public function/method/class needs a Google-style docstring (summary line +
  `Args:`/`Returns:`/`Raises:`) — Ruff's `pydocstyle` (`select = ["D"]`,
  `convention = "google"`) fails the build otherwise. Tests are exempt (see below).
- Every pytest test must use literal `# 1. ARRANGE`, `# 2. ACT`, `# 3. ASSERT` comment
  blocks separated by blank lines — this is a constitution requirement checked in review,
  not just style.
- New modules under `src/` need corresponding tests under `tests/`.
- MLflow is the system of record for experiment runs; any new run must carry a `spec_id` tag
  identifying which `specs/<NNN>-.../` feature produced it.
- Branching: `main` is protected (single-commit history) and `develop` is the integration
  branch — PRs target `develop`, not `main`. Work happens on feature/member branches; never
  commit directly to either `main` or `develop`. Commit messages follow
  `tipo: descripción breve` (`feat`, `fix`, `chore`, `docs`, `test`, `refactor`).
- Full rules used for AI-assisted code review live in `AGENTS.md`; the constitution at
  `.specify/memory/constitution.md` is the source of truth if the two ever disagree.
