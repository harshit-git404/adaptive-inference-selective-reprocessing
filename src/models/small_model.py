"""Lightweight baseline text classifier for the adaptive inference pipeline.

This module provides a small CPU-friendly model built with TF-IDF features and
logistic regression. It is intended to serve as the fast first-pass model in an
adaptive inference system:

1. The small model makes an initial prediction for every input.
2. Probability scores from the model are inspected to estimate uncertainty.
3. Only low-confidence examples or tokens are escalated to a larger model.
4. Outputs from both models can later be fused into a final prediction.

Probability output matters because adaptive inference depends on confidence
signals. Even when the predicted label is correct, the class probabilities help
the pipeline decide whether the prediction is certain enough to keep or whether
it should be refined by a larger model.

The implementation uses scikit-learn only and runs on CPU without any GPU
requirements.
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
    """Create the TF-IDF vectorizer for the small baseline model."""

    return TfidfVectorizer(
        max_features=int(_get_config_value(config, "SMALL_MODEL_MAX_FEATURES")),
        lowercase=True,
        ngram_range=(1, 1),
    )


def _build_classifier(config: Any) -> LogisticRegression:
    """Create the lightweight logistic regression classifier."""

    return LogisticRegression(
        max_iter=200,
        random_state=_get_config_value(config, "RANDOM_SEED"),
        solver="liblinear",
    )


def train_small_model(
    train_texts: list[str],
    train_labels: list[int],
    config: Any,
) -> tuple[LogisticRegression, TfidfVectorizer]:
    """Train the small baseline model used for the first inference pass.

    Args:
        train_texts: Input training sentences.
        train_labels: Integer sentiment labels aligned with ``train_texts``.
        config: Configuration object containing
            ``SMALL_MODEL_MAX_FEATURES`` and optionally ``RANDOM_SEED``.

    Returns:
        A tuple of ``(trained_model, vectorizer)`` ready for inference.

    This model is deliberately lightweight so it can produce fast initial
    predictions before the adaptive pipeline decides whether more expensive
    processing is necessary.
    """

    vectorizer = _build_vectorizer(config)
    features = vectorizer.fit_transform(train_texts)

    model = _build_classifier(config)
    model.fit(features, train_labels)
    return model, vectorizer


def predict_small(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    texts: list[str],
) -> list[int]:
    """Predict class labels with the trained small model.

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


def predict_proba_small(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    texts: list[str],
) -> list[list[float]]:
    """Predict per-class probabilities for adaptive inference decisions.

    Args:
        model: Trained logistic regression classifier.
        vectorizer: Fitted TF-IDF vectorizer.
        texts: Input texts to score.

    Returns:
        A list of probability distributions, one per input text.

    The probability scores are a key part of the adaptive inference pipeline
    because they provide a simple confidence signal. Future uncertainty modules
    can use these scores to decide which predictions should be escalated to a
    larger model.
    """

    features = vectorizer.transform(texts)
    probabilities = model.predict_proba(features)
    return probabilities.tolist()
