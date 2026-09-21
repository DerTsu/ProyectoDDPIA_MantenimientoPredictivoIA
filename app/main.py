"""Orquestador principal de la interfaz de mantenimiento predictivo."""

import sys
from pathlib import Path

# Streamlit ejecuta este archivo directamente, así que agrega `app/` al path y no la
# raíz del proyecto. Sin esto, los imports `app.*` de abajo fallan con ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.components.form import render_machine_form
from app.components.results import render_results
from app.components.settings import render_alert_sensitivity
from app.utils.model import (
    InferenceServiceUnavailableError,
    check_inference_health,
    predict_failure,
)
from app.utils.preprocessing import preprocess_input

st.set_page_config(
    page_title="Mantenimiento Predictivo",
    page_icon=":material/engineering:",
    layout="centered",
)

st.title("Mantenimiento Predictivo")
st.caption("Clasificación de fallas con TabPFN-v2 · Dataset AI4I 2020")

umbral_falla = render_alert_sensitivity()

if check_inference_health():
    st.caption("🟢 Servicio de inferencia (gRPC) conectado")
else:
    st.caption("🔴 Servicio de inferencia (gRPC) no disponible — la predicción fallará")

inputs = render_machine_form()

if inputs:
    with st.spinner("Ejecutando inferencia con TabPFN-v2..."):
        X_new = preprocess_input(
            inputs["product_type"],
            inputs["air_temp"],
            inputs["process_temp"],
            inputs["rpm"],
            inputs["torque"],
            inputs["tool_wear"],
        )
        try:
            proba = predict_failure(X_new)
        except InferenceServiceUnavailableError as exc:
            st.error(str(exc))
            st.stop()

    st.session_state.resultado = {
        "proba": proba,
        "air_temp": inputs["air_temp"],
        "process_temp": inputs["process_temp"],
        "torque": inputs["torque"],
        "rpm": inputs["rpm"],
        "X_new": X_new,
    }

if "resultado" in st.session_state:
    render_results(
        proba=st.session_state.resultado["proba"],
        air_temp=st.session_state.resultado["air_temp"],
        process_temp=st.session_state.resultado["process_temp"],
        torque=st.session_state.resultado["torque"],
        rpm=st.session_state.resultado["rpm"],
        X_new=st.session_state.resultado["X_new"],
        umbral_falla=umbral_falla,
    )
