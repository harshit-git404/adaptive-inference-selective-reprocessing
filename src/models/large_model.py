"""More expressive text classifier for selective adaptive inference.

This module provides the larger text model used in the deeper computation stage
of the adaptive inference pipeline. It is intended to run only on selectively
routed uncertain inputs after the lightweight small model has already produced
an initial prediction.

Compared with the baseline path, this stage uses a larger TF-IDF feature space
and word/phrase n-grams to capture slightly richer patterns in text while still
remaining CPU-friendly and easy to debug. The implementation uses scikit-learn
only and does not require any GPU support.
"""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _get_config_value(config: Any, key: str) -> Any:
    """Read a configuration value from either an object or a dictionary."""

    if isinstance(config, dict):
        return config[key]
    return getattr(config, key)


def _build_vectorizer(config: Any) -> TfidfVectorizer:
    """Create the TF-IDF vectorizer for the larger selective model.

    The larger model uses unigram and bigram features so it can capture short
    phrases and provide a richer representation than the small baseline model.
    """

    return TfidfVectorizer(
        max_features=int(_get_config_value(config, "LARGE_MODEL_MAX_FEATURES")),
        lowercase=True,
        ngram_range=(1, 2),
    )


def _build_classifier(config: Any) -> LogisticRegression:
    """Create the logistic regression classifier for the larger model."""

    return LogisticRegression(
        max_iter=500,
        random_state=_get_config_value(config, "RANDOM_SEED"),
        solver="liblinear",
    )


def train_large_model(
    train_texts: list[str],
    train_labels: list[int],
    config: Any,
) -> tuple[LogisticRegression, TfidfVectorizer]:
    """Train the larger model used for selectively routed uncertain inputs.

    Args:
        train_texts: Input training sentences.
        train_labels: Integer class labels aligned with ``train_texts``.
        config: Configuration object containing
            ``LARGE_MODEL_MAX_FEATURES`` and optionally ``RANDOM_SEED``.

    Returns:
        A tuple of ``(trained_model, vectorizer)`` ready for inference.

    This model represents the deeper computation stage of adaptive inference.
    It increases expressive capacity through unigram and bigram TF-IDF
    features, which helps it model short phrases in selectively reprocessed
    text.
    """

    vectorizer = _build_vectorizer(config)
    features = vectorizer.fit_transform(train_texts)

    model = _build_classifier(config)
    model.fit(features, train_labels)
    return model, vectorizer


def predict_large(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    texts: list[str],
) -> list[int]:
    """Predict class labels with the trained larger model.

    Args:
        model: Trained logistic regression classifier.
        vectorizer: Fitted TF-IDF vectorizer.
        texts: Input texts to classify.

    Returns:
        Predicted class labels.
    """

    features = vectorizer.transform(texts)
    predictions = model.predict(features)
    return predictions.tolist()


def predict_proba_large(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    texts: list[str],
) -> list[list[float]]:
    """Predict per-class probabilities for selectively routed inputs.

    Args:
        model: Trained logistic regression classifier.
        vectorizer: Fitted TF-IDF vectorizer.
        texts: Input texts to score.

    Returns:
        A list of probability distributions, one per input text.

    These probabilities allow the larger model to participate in downstream
    fusion and uncertainty-aware analysis after selective reprocessing.
    """

    features = vectorizer.transform(texts)
    probabilities = model.predict_proba(features)
    return probabilities.tolist()


class LargeModel:
    """Thin wrapper around the module-level large-model utilities."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.vectorizer: TfidfVectorizer | None = None

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        config: Any,
    ) -> tuple[LogisticRegression, TfidfVectorizer]:
        """Train and store the larger model components."""

        self.model, self.vectorizer = train_large_model(
            train_texts,
            train_labels,
            config,
        )
        return self.model, self.vectorizer

    def predict(self, texts: list[str]) -> list[int]:
        """Predict class labels using the stored model state."""

        if self.model is None or self.vectorizer is None:
            raise ValueError("LargeModel must be trained before prediction.")
        return predict_large(self.model, self.vectorizer, texts)

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        """Predict probabilities using the stored model state."""

        if self.model is None or self.vectorizer is None:
            raise ValueError("LargeModel must be trained before prediction.")
        return predict_proba_large(self.model, self.vectorizer, texts)

    def refine(self, texts: list[str]) -> list[int]:
        """Alias for prediction during selective reprocessing."""

        return self.predict(texts)
