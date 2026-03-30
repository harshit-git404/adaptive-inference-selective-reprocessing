"""Model implementations for adaptive inference experiments."""

from .large_model import LargeModel, predict_large, predict_proba_large, train_large_model
from .small_model import predict_proba_small, predict_small, train_small_model

__all__ = [
    "LargeModel",
    "train_large_model",
    "predict_large",
    "predict_proba_large",
    "train_small_model",
    "predict_small",
    "predict_proba_small",
]
