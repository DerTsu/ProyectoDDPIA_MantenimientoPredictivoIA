"""T009 [US1] Tests unitarios para load_raw_data()."""

import pandas as pd
import pytest

from src.preprocessing import load_raw_data

COLUMNAS_ESPERADAS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def test_retorna_dataframe_con_10000_registros(dataset_path):
    # 1. ARRANGE (Ruta al CSV crudo de AI4I 2020)
    ruta = dataset_path

    # 2. ACT (Cargar el dataset)
    df = load_raw_data(ruta)

    # 3. ASSERT (Verificar tipo y número de registros)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 10000


def test_columnas_esperadas_existen(dataset_path):
    # 1. ARRANGE (Ruta al CSV crudo)
    ruta = dataset_path

    # 2. ACT (Cargar el dataset)
    df = load_raw_data(ruta)

    # 3. ASSERT (Verificar que están las 14 columnas originales en orden)
    assert list(df.columns) == COLUMNAS_ESPERADAS


def test_sin_valores_faltantes(dataset_path):
    # 1. ARRANGE (Ruta al CSV crudo)
    ruta = dataset_path

    # 2. ACT (Cargar el dataset y contar nulos)
    n_nulos = load_raw_data(ruta).isna().sum().sum()

    # 3. ASSERT (El dataset AI4I 2020 se asume limpio)
    assert n_nulos == 0


def test_target_es_binario(dataset_path):
    # 1. ARRANGE (Ruta al CSV crudo)
    ruta = dataset_path

    # 2. ACT (Cargar y extraer los valores únicos del target)
    valores = set(load_raw_data(ruta)["Machine failure"].unique())

    # 3. ASSERT (Solo debe contener 0 y 1)
    assert valores <= {0, 1}


def test_type_tiene_tres_categorias(dataset_path):
    # 1. ARRANGE (Ruta al CSV crudo)
    ruta = dataset_path

    # 2. ACT (Cargar y extraer las categorías de Type)
    categorias = set(load_raw_data(ruta)["Type"].unique())

    # 3. ASSERT (Los tres tipos de producto documentados)
    assert categorias == {"L", "M", "H"}


def test_archivo_inexistente_lanza_error(tmp_path):
    # 1. ARRANGE (Ruta a un archivo que no existe)
    ruta = tmp_path / "no_existe.csv"

    # 2. ACT + 3. ASSERT (Cargar debe fallar con un error explícito)
    with pytest.raises(FileNotFoundError):
        load_raw_data(ruta)


def test_dataset_con_nulos_lanza_error(dataset_path, tmp_path):
    # 1. ARRANGE (Copia del dataset con un valor faltante inyectado)
    df = load_raw_data(dataset_path)
    df.loc[0, "Torque [Nm]"] = None
    ruta_corrupta = tmp_path / "con_nulos.csv"
    df.to_csv(ruta_corrupta, index=False)

    # 2. ACT + 3. ASSERT (La carga debe rechazarlo antes de modelar)
    with pytest.raises(ValueError, match="valores faltantes"):
        load_raw_data(ruta_corrupta)
