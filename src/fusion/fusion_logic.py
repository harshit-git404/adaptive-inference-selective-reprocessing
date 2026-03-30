"""Probability fusion utilities for adaptive inference.

Weighted fusion combines the fast small-model prediction with the more
selective large-model prediction. This supports the adaptive inference
architecture by preserving the efficiency of the first-pass model while still
allowing uncertain cases to benefit from deeper computation.
"""

from __future__ import annotations

from typing import Any


def _get_config_value(config: Any, key: str) -> float:
    """Read a configuration value from either an object or a dictionary."""

    if isinstance(config, dict):
        return float(config[key])
    return float(getattr(config, key))


def normalize_probability(prob) -> float:
    """Return a safe probability clipped into the inclusive [0, 1] range.

    ``None`` is treated as ``0.0`` so fusion can proceed without raising on
    missing values. Numeric values outside the valid probability range are
    clipped to keep downstream behavior stable and easy to debug.
    """

    if prob is None:
        return 0.0

    try:
        numeric_prob = float(prob)
    except (TypeError, ValueError) as exc:
        raise TypeError("Probability values must be numeric or None.") from exc

    return max(0.0, min(1.0, numeric_prob))


def fuse_predictions(
    prob_small: float,
    prob_large: float,
    config,
) -> dict[str, float | int]:
    """Fuse small-model and large-model probabilities into one prediction.

    Weighted fusion is used so the adaptive inference system can retain the
    small model's broad first-pass signal while incorporating the large model's
    more expressive reprocessing result for uncertain inputs.

    Args:
        prob_small: Positive-class probability from the small model.
        prob_large: Positive-class probability from the large model.
        config: Configuration object or dictionary containing fusion weights.

    Returns:
        A dictionary with the fused probability and final binary label.
    """

    # Normalize inputs first so missing or slightly invalid values do not
    # propagate confusing behavior into the fusion step.
    normalized_small = normalize_probability(prob_small)
    normalized_large = normalize_probability(prob_large)

    # Read weights from config so the fusion behavior is easy to tune.
    weight_small = _get_config_value(config, "FUSION_WEIGHT_SMALL")
    weight_large = _get_config_value(config, "FUSION_WEIGHT_LARGE")

    # Compute the final weighted probability used for the binary decision.
    final_probability = (
        weight_small * normalized_small +
        weight_large * normalized_large
    )
    final_probability = normalize_probability(final_probability)

    # Convert the fused score into the final class label.
    final_label = 1 if final_probability >= 0.5 else 0

    return {
        "final_probability": final_probability,
        "final_label": final_label,
    }


class FusionLogic:
    """Thin wrapper around the module-level fusion helpers."""

    def __init__(self, config) -> None:
        self.config = config

    def combine(self, prob_small: float, prob_large: float) -> dict[str, float | int]:
        """Fuse two probabilities using the stored configuration."""

        return fuse_predictions(prob_small, prob_large, self.config)
