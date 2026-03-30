"""Token-level uncertainty estimation for adaptive inference.

This module uses a simple representation-instability heuristic:

1. Compute the TF-IDF vector for the full sentence.
2. Remove one token at a time using whitespace tokenization.
3. Recompute the TF-IDF vector after each token removal.
4. Measure the L2 distance between the original and modified vectors.

Tokens that cause a larger feature-space shift are treated as more uncertain or
more influential for the current prediction. The implementation is
intentionally simple and easy to debug, with no complex NLP preprocessing.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _get_threshold(config: Any) -> float:
    """Return the uncertainty threshold from either an object or dictionary."""

    if isinstance(config, dict):
        return float(config["UNCERTAINTY_THRESHOLD"])
    return float(config.UNCERTAINTY_THRESHOLD)


def remove_token(sentence: str, token_index: int) -> str:
    """Return the sentence with the token at ``token_index`` removed.

    The function uses simple whitespace tokenization to keep behavior
    deterministic and easy to inspect. If the index is invalid or the sentence
    has no tokens, the original normalized sentence is returned unchanged.
    """

    tokens = sentence.split()
    if not tokens:
        return ""
    if token_index < 0 or token_index >= len(tokens):
        return " ".join(tokens)

    remaining_tokens = tokens[:token_index] + tokens[token_index + 1 :]
    return " ".join(remaining_tokens)


def compute_token_uncertainty(
    model,
    vectorizer,
    sentence: str,
    config,
) -> list[dict[str, str | int | float]]:
    """Compute token uncertainty scores via leave-one-token-out analysis.

    Args:
        model: Trained small classification model.
        vectorizer: Fitted vectorizer used by the small model.
        sentence: Input sentence to analyze.
        config: Configuration object or dictionary containing
            ``UNCERTAINTY_THRESHOLD``.

    Returns:
        A list of dictionaries, one per token, each containing the token text,
        its position, and the uncertainty score defined as the L2 distance
        between the original sentence vector and the vector after removing that
        token.
    """

    tokens = sentence.split()
    if not tokens:
        return []

    base_vector = vectorizer.transform([sentence]).toarray()[0]

    token_scores: list[dict[str, str | int | float]] = []
    for index, token in enumerate(tokens):
        modified_sentence = remove_token(sentence, index)
        modified_vector = vectorizer.transform([modified_sentence]).toarray()[0]

        # Representation instability computed as embedding distance
        # approximates change in intermediate feature representation.
        uncertainty_score = float(np.linalg.norm(base_vector - modified_vector))

        token_scores.append(
            {
                "token": token,
                "index": index,
                "uncertainty_score": uncertainty_score,
            }
        )

    return token_scores


def identify_uncertain_tokens(
    model,
    vectorizer,
    sentence: str,
    config,
) -> list[int]:
    """Return token indices whose uncertainty exceeds the configured threshold.

    This uses the scores from :func:`compute_token_uncertainty` and compares
    them against ``config.UNCERTAINTY_THRESHOLD``. For a single-word sentence,
    the lone token is always returned.
    """

    token_scores = compute_token_uncertainty(model, vectorizer, sentence, config)
    if len(token_scores) == 1:
        return [0]

    threshold = _get_threshold(config)
    return [
        int(token_info["index"])
        for token_info in token_scores
        if float(token_info["uncertainty_score"]) > threshold
    ]


class UncertaintyEstimator:
    """Thin wrapper around the module-level uncertainty utilities."""

    def __init__(self, config) -> None:
        self.config = config

    def score(self, model, vectorizer, sentence: str):
        """Return per-token uncertainty details for ``sentence``."""

        return compute_token_uncertainty(model, vectorizer, sentence, self.config)

    def identify(self, model, vectorizer, sentence: str) -> list[int]:
        """Return uncertain token indices for ``sentence``."""

        return identify_uncertain_tokens(model, vectorizer, sentence, self.config)
