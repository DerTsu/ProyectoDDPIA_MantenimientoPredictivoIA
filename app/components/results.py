"""Renderizado de resultados: alertas, métricas y detalles técnicos."""

import numpy as np
import streamlit as st

from app.utils.physics import (
    calculate_delta_t,
    calculate_power,
    get_alert_level,
)
from app.utils.preprocessing import NOMBRE_FEATURES


def render_results(
    proba: np.ndarray,
    air_temp: float,
    process_temp: float,
    torque: float,
    rpm: float,
    X_new: np.ndarray,
    umbral_falla: float,
) -> None:
    """Renderiza la sección de resultados tras la inferencia.

    Muestra el semáforo de alerta, métricas físicas derivadas y detalles
    técnicos de la predicción en expanders.

    Args:
        proba: Matriz de probabilidades `(1, 2)` del modelo.
        air_temp: Temperatura del aire ingresada.
        process_temp: Temperatura del proceso ingresada.
        torque: Torque ingresado.
        rpm: Velocidad de rotación ingresada.
        X_new: Vector de features `(1, 8)` enviado al modelo.
        umbral_falla: Umbral de decisión de Falla inminente, configurado en la
            barra lateral por `render_alert_sensitivity`.
    """
    prob_falla = float(proba[0, 1])

    alerta = get_alert_level(prob_falla, umbral_falla)
    potencia = calculate_power(torque, rpm)
    delta_t = calculate_delta_t(process_temp, air_temp)

    st.subheader("Resultado")

    if alerta == "Normal":
        st.success(f"**{alerta}** — Probabilidad de falla: {prob_falla:.1%}")
    elif alerta == "Precaucion":
        st.warning(f"**{alerta}** — Probabilidad de falla: {prob_falla:.1%}")
    else:
        st.error(f"**{alerta}** — Probabilidad de falla: {prob_falla:.1%}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ΔT (Proceso − Aire)", f"{delta_t:.1f} K")
    with col2:
        st.metric("Potencia mecánica", f"{potencia:.0f} W")
    with col3:
        st.metric("Prob. de falla", f"{prob_falla:.1%}")

    with st.expander("Detalles de la predicción"):
        st.write("**Vector de entrada (8 features):**")
        st.code(
            dict(zip(NOMBRE_FEATURES, X_new[0], strict=False)),
            language="json",
        )
        st.write(
            f"**Probabilidades completas:** No falla {proba[0, 0]:.4f} · Falla {proba[0, 1]:.4f}"
        )

    with st.expander("Información del modelo TabPFN-v2"):
        st.markdown(
            "- **Arquitectura:** Prior-data Fitted Network"
            " (In-Context Learning con Transformer).\n"
            "- **Modo de inferencia:** Inferencia directa sin ajuste por gradiente.\n"
            "- **Muestras en contexto:** 1,000 registros estratificados"
            " (optimizado para CPU)."
        )
