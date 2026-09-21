"""Servicio de inferencia gRPC: expone TabPFN-v2 en un proceso propio.

Separa el modelo (y su stack pesado: torch, tabpfn) de la interfaz Streamlit
en `app/`, que ahora es un cliente gRPC ligero (ver `app/utils/model.py`).
"""
