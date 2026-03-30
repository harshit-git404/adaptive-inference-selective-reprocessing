"""Synthetic sentiment dataset generation utilities.

This module creates a small balanced dataset for experimentation without any
external downloads. The generated sentences use simple templates with light
wording variations so the data is easy to inspect and debug.
"""

from __future__ import annotations

import random

from config.config import Config

POSITIVE_TEMPLATES = [
    "I love this product",
    "This movie was amazing",
    "The service was excellent",
]

NEGATIVE_TEMPLATES = [
    "I hate this product",
    "This movie was terrible",
    "The service was disappointing",
]

PHRASE_VARIATIONS = {
    "I love this product": [
        "I really love this product",
        "I absolutely love this product",
        "I genuinely like this product",
        "I love this product a lot",
    ],
    "This movie was amazing": [
        "This movie was really amazing",
        "This movie was fantastic",
        "This movie was incredibly good",
        "This movie was wonderful",
    ],
    "The service was excellent": [
        "The service was truly excellent",
        "The service was outstanding",
        "The service was very good",
        "The service was impressive",
    ],
    "I hate this product": [
        "I really hate this product",
        "I absolutely dislike this product",
        "I strongly dislike this product",
        "I hate this product a lot",
    ],
    "This movie was terrible": [
        "This movie was really terrible",
        "This movie was awful",
        "This movie was incredibly bad",
        "This movie was disappointing",
    ],
    "The service was disappointing": [
        "The service was very disappointing",
        "The service was poor",
        "The service was frustrating",
        "The service was far below expectations",
    ],
}

PREFIXES = [
    "",
    "Honestly, ",
    "In my opinion, ",
    "Overall, ",
]

SUFFIXES = [
    "",
    " today.",
    " overall.",
    " for me.",
]


def _vary_sentence(template: str, rng: random.Random) -> str:
    """Return a lightly varied version of a base template."""

    base_sentence = rng.choice([template, *PHRASE_VARIATIONS[template]])
    prefix = rng.choice(PREFIXES)
    suffix = rng.choice(SUFFIXES)
    sentence = f"{prefix}{base_sentence}{suffix}".strip()

    if sentence.endswith("."):
        return sentence
    return f"{sentence}."


def generate_dataset(num_samples: int = 300) -> tuple[list[str], list[int]]:
    """Generate a balanced synthetic sentiment dataset.

    Args:
        num_samples: Total number of sentences to generate. The value is clamped
            into the 200-400 range and adjusted to remain even for balance.

    Returns:
        A tuple of ``(texts, labels)`` where labels use ``1`` for positive and
        ``0`` for negative sentiment.
    """

    bounded_samples = max(200, min(400, num_samples))
    if bounded_samples % 2 != 0:
        bounded_samples += 1

    rng = random.Random(Config.RANDOM_SEED)
    samples_per_class = bounded_samples // 2

    texts: list[str] = []
    labels: list[int] = []

    for _ in range(samples_per_class):
        texts.append(_vary_sentence(rng.choice(POSITIVE_TEMPLATES), rng))
        labels.append(1)

    for _ in range(samples_per_class):
        texts.append(_vary_sentence(rng.choice(NEGATIVE_TEMPLATES), rng))
        labels.append(0)

    paired_examples = list(zip(texts, labels))
    rng.shuffle(paired_examples)

    shuffled_texts = [text for text, _ in paired_examples]
    shuffled_labels = [label for _, label in paired_examples]
    return shuffled_texts, shuffled_labels


def load_real_dataset(sample_size: int = 300) -> tuple[list[str], list[int]]:
    """Load a deterministic sample of real sentiment data from IMDb.

    Args:
        sample_size: Number of examples to sample from the IMDb training split.

    Returns:
        A tuple of ``(texts, labels)`` sampled from the dataset.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for DATASET_TYPE='imdb'. "
            "Install it with 'pip install -r requirements.txt'."
        ) from exc

    dataset = load_dataset("imdb", split="train")
    rng = random.Random(Config.RANDOM_SEED)

    bounded_sample_size = max(1, min(sample_size, len(dataset)))
    selected_indices = rng.sample(range(len(dataset)), bounded_sample_size)

    texts = [str(dataset[index]["text"]).strip() for index in selected_indices]
    labels = [int(dataset[index]["label"]) for index in selected_indices]
    return texts, labels


def split_dataset(
    texts: list[str] | None = None,
    labels: list[int] | None = None,
    test_ratio: float = 0.2,
    num_samples: int = 300,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Generate the dataset and split it into train and test sets.

    Args:
        texts: Optional pre-generated input texts to split.
        labels: Optional labels aligned with ``texts``.
        test_ratio: Fraction of samples reserved for testing.
        num_samples: Total number of examples to generate before splitting when
            ``texts`` and ``labels`` are not provided.

    Returns:
        ``train_texts, train_labels, test_texts, test_labels``
    """

    if texts is None or labels is None:
        texts, labels = generate_dataset(num_samples=num_samples)
    elif len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length.")

    rng = random.Random(Config.RANDOM_SEED)

    positive_examples = [
        (text, label) for text, label in zip(texts, labels) if label == 1
    ]
    negative_examples = [
        (text, label) for text, label in zip(texts, labels) if label == 0
    ]

    rng.shuffle(positive_examples)
    rng.shuffle(negative_examples)

    positive_test_count = int(len(positive_examples) * test_ratio)
    negative_test_count = int(len(negative_examples) * test_ratio)

    positive_test = positive_examples[:positive_test_count]
    positive_train = positive_examples[positive_test_count:]
    negative_test = negative_examples[:negative_test_count]
    negative_train = negative_examples[negative_test_count:]

    train_examples = positive_train + negative_train
    test_examples = positive_test + negative_test

    rng.shuffle(train_examples)
    rng.shuffle(test_examples)

    train_texts = [text for text, _ in train_examples]
    train_labels = [label for _, label in train_examples]
    test_texts = [text for text, _ in test_examples]
    test_labels = [label for _, label in test_examples]

    return train_texts, train_labels, test_texts, test_labels
