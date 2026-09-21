"""Carga del modelo TabPFN-v2 y su contexto de referencia para el servicio gRPC.

Equivalente sin Streamlit de la antigua `app/utils/model.py`: el servicio de
inferencia corre en su propio proceso (contenedor `inference`), separado de la
interfaz web, así que el contexto se ajusta una sola vez al arrancar el
servidor en vez de depender de `st.cache_resource`.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.models import TabPFNClassifier
from src.preprocessing import build_features, load_raw_data

DATASET_PATH = Path("data/raw/ai4i2020.csv")
N_CONTEXTO_SERVICIO = 1000


def load_model_and_context(
    dataset_path: Path = DATASET_PATH,
    n_samples: int = N_CONTEXTO_SERVICIO,
) -> tuple[TabPFNClassifier, np.ndarray, np.ndarray]:
    """Carga el dataset y prepara un contexto estratificado optimizado para CPU.

    Args:
        dataset_path: Ruta al CSV crudo de AI4I 2020.
        n_samples: Número de muestras de referencia a usar en contexto (1000 por
            defecto, igual que en la interfaz web, para latencia consistente).

    Returns:
        Tupla `(modelo, X_ctx, y_ctx)` lista para inferencia vía gRPC.
    """
    df = load_raw_data(dataset_path)
    X_train, y_train, _, _ = build_features(df)

    if len(X_train) > n_samples:
        X_ctx, _, y_ctx, _ = train_test_split(
            X_train,
            y_train,
            train_size=n_samples,
            stratify=y_train,
            random_state=42,
        )
    else:
        X_ctx, y_ctx = X_train, y_train

    model = TabPFNClassifier()
    model.fit_context(X_ctx, y_ctx)
    return model, X_ctx, y_ctx
