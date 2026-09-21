"""Modelos de clasificación de fallos: TabPFN-v2 (base) y XGBoost (baseline)."""

from src.models.tabpfn_model import TabPFNClassifier
from src.models.xgboost_baseline import XGBoostBaseline

__all__ = ["TabPFNClassifier", "XGBoostBaseline"]
