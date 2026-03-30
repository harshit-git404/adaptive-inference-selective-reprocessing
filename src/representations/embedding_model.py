"""Sentence embedding utilities for adaptive inference experiments.

This module exposes a reusable sentence embedding function backed by a
transformer encoder. The embeddings are used to compute semantic
representations for uncertainty estimation.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
except Exception:
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def get_sentence_embedding(sentence: str) -> np.ndarray:
    """Return a deterministic sentence embedding as a NumPy array.

    The model is loaded once at module import time and reused for all calls, so
    repeated uncertainty estimation stays consistent and avoids reload overhead.
    """

    embedding = _MODEL.encode(
        sentence,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(embedding, dtype=float)
