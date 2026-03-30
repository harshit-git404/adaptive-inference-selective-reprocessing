"""Selective routing utilities for adaptive inference.

This module reduces an input sentence to only the tokens marked as uncertain.
The reduced text can then be passed to a larger model for targeted
reprocessing, while the full sentence is kept for context and debugging.
"""

from __future__ import annotations


def _normalize_indices(tokens: list[str], uncertain_indices: list[int]) -> list[int]:
    """Return unique, valid indices sorted in original token order."""

    valid_indices = {
        index
        for index in uncertain_indices
        if isinstance(index, int) and 0 <= index < len(tokens)
    }
    return [index for index in range(len(tokens)) if index in valid_indices]


def extract_uncertain_subsequence(sentence: str, uncertain_indices: list[int]) -> str:
    """Extract a reduced sentence containing only uncertain tokens.

    Args:
        sentence: Original input sentence.
        uncertain_indices: Token indices marked as uncertain.

    Returns:
        A whitespace-joined string containing only the uncertain tokens in their
        original order. If no valid uncertain tokens are found, the original
        sentence is returned unchanged.
    """

    # Use simple whitespace tokenization so behavior stays deterministic.
    tokens = sentence.split()
    if not tokens:
        return sentence

    # Remove duplicates and ignore invalid indices.
    normalized_indices = _normalize_indices(tokens, uncertain_indices)
    if not normalized_indices:
        return sentence

    # Collect the uncertain tokens while preserving the sentence order.
    selected_tokens = [tokens[index] for index in normalized_indices]
    return " ".join(selected_tokens)


def route_for_reprocessing(sentence: str, uncertain_indices: list[int]) -> dict:
    """Prepare routing metadata for selective reprocessing.

    Args:
        sentence: Original input sentence.
        uncertain_indices: Token indices considered uncertain.

    Returns:
        A dictionary containing the original sentence, the selected uncertain
        tokens, the reduced sentence, and the number of uncertain tokens.
    """

    # Tokenize once so the routing summary is easy to inspect in a debugger.
    tokens = sentence.split()
    normalized_indices = _normalize_indices(tokens, uncertain_indices)

    # Extract uncertain tokens in their original order.
    uncertain_tokens = [tokens[index] for index in normalized_indices]

    # Fall back to the full sentence when nothing valid was selected.
    reduced_sentence = (
        " ".join(uncertain_tokens) if uncertain_tokens else sentence
    )

    return {
        "original_sentence": sentence,
        "uncertain_tokens": uncertain_tokens,
        "reduced_sentence": reduced_sentence,
        "num_uncertain_tokens": len(uncertain_tokens),
    }


class SelectiveRouter:
    """Thin wrapper around the routing helper functions."""

    def extract(self, sentence: str, uncertain_indices: list[int]) -> str:
        """Return the reduced sentence containing only uncertain tokens."""

        return extract_uncertain_subsequence(sentence, uncertain_indices)

    def select(self, sentence: str, uncertain_indices: list[int]) -> dict:
        """Return routing details for selective reprocessing."""

        return route_for_reprocessing(sentence, uncertain_indices)
