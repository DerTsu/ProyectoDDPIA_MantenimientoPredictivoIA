"""Analisis Exploratorio de Datos (EDA) - Dataset AI4I 2020 Predictive Maintenance.

Genera tablas y graficas descriptivas del dataset usado en el proyecto de
mantenimiento predictivo (Modulo 2, Sprint 2), guardandolas en EDA/ para
su uso en el informe.

Uso:
    uv run python EDA/eda.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/raw/ai4i2020.csv")
OUTPUT_DIR = Path("EDA")
FIGURES_DIR = OUTPUT_DIR / "figures"

PROCESS_VARS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carga el dataset AI4I 2020 desde un archivo CSV.

    Args:
        path: Ruta al archivo CSV del dataset.

    Returns:
        DataFrame con los datos crudos.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset en {path}. Verifica que data/raw/ai4i2020.csv exista."
        )
    return pd.read_csv(path)


def save_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    """Calcula y guarda las tablas resumen del EDA como CSV.

    Genera: estadisticas descriptivas, balance de clases (con/sin falla),
    conteo por modo de falla y conteo por tipo de producto.

    Args:
        df: DataFrame con los datos crudos.
        output_dir: Carpeta donde se guardan los CSV resultantes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    describe_df = df[PROCESS_VARS].describe().T
    describe_df.to_csv(output_dir / "estadisticas_descriptivas.csv")

    total = len(df)
    failure_counts = df["Machine failure"].value_counts().rename({0: "Sin falla", 1: "Con falla"})
    failure_pct = (failure_counts / total * 100).round(2)
    balance_df = pd.DataFrame({"conteo": failure_counts, "porcentaje": failure_pct})
    balance_df.to_csv(output_dir / "balance_clases.csv")

    mode_counts = df[FAILURE_MODES].sum().sort_values(ascending=False)
    mode_counts.to_csv(output_dir / "conteo_modos_falla.csv", header=["conteo"])

    type_counts = df["Type"].value_counts()
    type_counts.to_csv(output_dir / "conteo_tipo_producto.csv", header=["conteo"])


def plot_class_balance(df: pd.DataFrame, figures_dir: Path) -> None:
    """Grafica el balance de clases (registros con falla vs sin falla).

    Args:
        df: DataFrame con los datos crudos.
        figures_dir: Carpeta donde se guarda la figura generada.
    """
    counts = df["Machine failure"].value_counts().sort_index()
    labels = ["Sin falla", "Con falla"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts.values, color=["#2a9d8f", "#e76f51"])
    ax.set_title("Balance de clases: Machine failure")
    ax.set_ylabel("Numero de registros")
    ax.set_ylim(0, counts.values.max() * 1.15)
    for bar, count in zip(bars, counts.values, strict=True):
        pct = count / len(df) * 100
        ax.annotate(
            f"{count}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(figures_dir / "balance_clases.png", dpi=150)
    plt.close(fig)


def plot_failure_modes(df: pd.DataFrame, figures_dir: Path) -> None:
    """Grafica el conteo de registros por cada modo de falla.

    Args:
        df: DataFrame con los datos crudos.
        figures_dir: Carpeta donde se guarda la figura generada.
    """
    mode_counts = df[FAILURE_MODES].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(mode_counts.index, mode_counts.values, color="#457b9d")
    ax.set_title("Conteo de registros por modo de falla")
    ax.set_ylabel("Numero de registros")
    fig.tight_layout()
    fig.savefig(figures_dir / "modos_falla.png", dpi=150)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, figures_dir: Path) -> None:
    """Grafica histogramas de las variables de proceso.

    Args:
        df: DataFrame con los datos crudos.
        figures_dir: Carpeta donde se guarda la figura generada.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, PROCESS_VARS, strict=False):
        ax.hist(df[col], bins=30, color="#264653", edgecolor="white")
        ax.set_title(col)

    for ax in axes[len(PROCESS_VARS) :]:
        ax.axis("off")

    fig.suptitle("Distribucion de variables de proceso")
    fig.tight_layout()
    fig.savefig(figures_dir / "distribuciones.png", dpi=150)
    plt.close(fig)


def plot_boxplots_by_failure(df: pd.DataFrame, figures_dir: Path) -> None:
    """Compara las variables de proceso entre registros con y sin falla.

    Args:
        df: DataFrame con los datos crudos.
        figures_dir: Carpeta donde se guarda la figura generada.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, PROCESS_VARS, strict=False):
        sin_falla = df.loc[df["Machine failure"] == 0, col]
        con_falla = df.loc[df["Machine failure"] == 1, col]
        ax.boxplot([sin_falla, con_falla], tick_labels=["Sin falla", "Con falla"])
        ax.set_title(col)

    for ax in axes[len(PROCESS_VARS) :]:
        ax.axis("off")

    fig.suptitle("Variables de proceso segun ocurrencia de falla")
    fig.tight_layout()
    fig.savefig(figures_dir / "boxplots_por_falla.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, figures_dir: Path) -> None:
    """Grafica un mapa de calor de correlaciones entre variables de proceso.

    Args:
        df: DataFrame con los datos crudos.
        figures_dir: Carpeta donde se guarda la figura generada.
    """
    corr = df[PROCESS_VARS].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax, label="Correlacion")
    ax.set_title("Correlacion entre variables de proceso")
    fig.tight_layout()
    fig.savefig(figures_dir / "correlaciones.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Ejecuta el pipeline completo de EDA y guarda tablas y graficas."""
    df = load_data()

    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Valores nulos totales: {df.isnull().sum().sum()}")

    save_summary_tables(df, OUTPUT_DIR)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_class_balance(df, FIGURES_DIR)
    plot_failure_modes(df, FIGURES_DIR)
    plot_distributions(df, FIGURES_DIR)
    plot_boxplots_by_failure(df, FIGURES_DIR)
    plot_correlation_heatmap(df, FIGURES_DIR)

    print(f"Tablas guardadas en: {OUTPUT_DIR}")
    print(f"Graficas guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
