"""Project configuration for adaptive inference experiments.

The parameters are defined as class attributes so they are easy to inspect and
modify in one place.

Attributes:
    SMALL_MODEL_MAX_FEATURES:
        Maximum number of features used by the lightweight first-pass model.
    LARGE_MODEL_MAX_FEATURES:
        Maximum number of features used by the larger fallback model.
    UNCERTAINTY_THRESHOLD:
        Threshold used to mark predictions as uncertain and route them for
        additional processing.
    FUSION_WEIGHT_SMALL:
        Contribution weight assigned to the small model during output fusion.
    FUSION_WEIGHT_LARGE:
        Contribution weight assigned to the large model during output fusion.
    RANDOM_SEED:
        Seed value used to keep experiments deterministic when randomness is
        introduced later.
"""


class Config:
    """Container for experiment-wide configuration values."""

    SMALL_MODEL_MAX_FEATURES = 1000
    LARGE_MODEL_MAX_FEATURES = 2000
    # Distance-based uncertainty scores operate on a larger numeric scale, so a
    # higher threshold is needed to preserve selective computation behavior.
    UNCERTAINTY_THRESHOLD = 0.60
    FUSION_WEIGHT_SMALL = 0.7
    FUSION_WEIGHT_LARGE = 0.3
    RANDOM_SEED = 42


def get_config() -> dict[str, float | int]:
    """Return the current configuration as a plain dictionary."""

    return {
        "SMALL_MODEL_MAX_FEATURES": Config.SMALL_MODEL_MAX_FEATURES,
        "LARGE_MODEL_MAX_FEATURES": Config.LARGE_MODEL_MAX_FEATURES,
        "UNCERTAINTY_THRESHOLD": Config.UNCERTAINTY_THRESHOLD,
        "FUSION_WEIGHT_SMALL": Config.FUSION_WEIGHT_SMALL,
        "FUSION_WEIGHT_LARGE": Config.FUSION_WEIGHT_LARGE,
        "RANDOM_SEED": Config.RANDOM_SEED,
    }
