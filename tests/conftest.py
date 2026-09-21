"""Fixtures compartidas para los tests de la feature 001."""

from pathlib import Path

import pytest

from src.preprocessing import build_features, load_raw_data

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "ai4i2020.csv"


@pytest.fixture(scope="session")
def dataset_path() -> Path:
    if not DATASET_PATH.exists():
        pytest.skip(f"Dataset no encontrado en {DATASET_PATH}")
    return DATASET_PATH


@pytest.fixture(scope="session")
def raw_df(dataset_path):
    return load_raw_data(dataset_path)


@pytest.fixture(scope="session")
def split_data(raw_df):
    return build_features(raw_df)


@pytest.fixture(scope="session")
def tabpfn_proba(split_data):
    """Probabilidades de TabPFN-v2 sobre X_test. Costosa: se calcula una sola vez."""
    from tabpfn.errors import TabPFNLicenseError

    from src.models import TabPFNClassifier

    X_train, y_train, X_test, _ = split_data
    try:
        return TabPFNClassifier().predict_proba(X_train, y_train, X_test)
    except TabPFNLicenseError:
        pytest.skip("Licencia de TabPFN no aceptada: ver https://ux.priorlabs.ai/account/licenses")


@pytest.fixture(scope="session")
def xgboost_proba(split_data):
    """Probabilidades de XGBoost sobre X_test. Se calcula una sola vez."""
    from src.models import XGBoostBaseline

    X_train, y_train, X_test, _ = split_data
    modelo = XGBoostBaseline.from_class_balance(y_train)
    modelo.fit(X_train, y_train)
    return modelo.predict_proba(X_test)


@pytest.fixture(scope="session")
def inference_servicer(dataset_path):
    """Servicer gRPC real, con TabPFN-v2 ya cargado. Costosa: una sola vez por sesión."""
    from tabpfn.errors import TabPFNLicenseError

    from src.serving.server import InferenceServicer

    del dataset_path  # Solo se usa para heredar el skip si falta el dataset.
    try:
        return InferenceServicer()
    except TabPFNLicenseError:
        pytest.skip("Licencia de TabPFN no aceptada: ver https://ux.priorlabs.ai/account/licenses")
