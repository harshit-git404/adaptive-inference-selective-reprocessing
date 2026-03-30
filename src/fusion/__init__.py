"""Fusion logic for combining adaptive inference outputs."""

from .fusion_logic import FusionLogic, fuse_predictions, normalize_probability

__all__ = ["FusionLogic", "fuse_predictions", "normalize_probability"]
