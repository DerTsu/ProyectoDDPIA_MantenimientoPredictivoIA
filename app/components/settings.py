"""Controles de configuración global de la interfaz."""

import streamlit as st

from app.utils.physics import (
    PASO_UMBRAL_FALLA,
    UMBRAL_FALLA,
    UMBRAL_FALLA_MAX,
    UMBRAL_FALLA_MIN,
)


def render_alert_sensitivity() -> float:
    """Renderiza en la barra lateral el slider de sensibilidad de alerta.

    Configura el umbral de decisión de Falla inminente, visible desde que se
    abre la app (no depende de haber hecho una predicción).

    Returns:
        Umbral de Falla inminente como fracción en `[0, 1]`.
    """
    with st.sidebar:
        umbral_falla_pct = st.slider(
            "Sensibilidad de alerta",
            min_value=round(UMBRAL_FALLA_MIN * 100),
            max_value=round(UMBRAL_FALLA_MAX * 100),
            value=round(UMBRAL_FALLA * 100),
            step=round(PASO_UMBRAL_FALLA * 100),
            format="%d%%",
            key="umbral_sensibilidad_alerta",
        )
        col_prev, col_cons = st.columns(2)
        col_prev.caption("◀ Más preventiva")
        col_cons.caption("Más conservadora ▶")

        umbral_falla = umbral_falla_pct / 100
        st.caption(
            f"**Umbral de decisión:** {umbral_falla:.0%}. Mayor sensibilidad"
            " detecta fallas con menor probabilidad, pero puede generar más falsas"
            " alarmas; menor sensibilidad reduce las alarmas, pero puede no detectar"
            " algunas fallas."
        )

    return umbral_falla
