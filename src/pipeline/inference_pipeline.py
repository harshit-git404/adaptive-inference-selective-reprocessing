"""End-to-end adaptive inference pipeline orchestration.

The pipeline connects all major stages of the demo:

1. Load configuration values.
2. Generate and split the synthetic dataset.
3. Train the small and large text classifiers.
4. Run the small model on each test sentence.
5. Identify uncertain tokens using leave-one-token-out stability analysis.
6. Route only the uncertain tokens to the large model.
7. Fuse small-model and large-model probabilities into a final prediction.

The implementation is intentionally straightforward so each stage can be
inspected independently during experimentation and debugging.
"""

from __future__ import annotations

from config.config import get_config
from data.data_loader import generate_dataset, split_dataset
from src.fusion.fusion_logic import fuse_predictions
from src.models.large_model import predict_proba_large, train_large_model
from src.models.small_model import predict_proba_small, train_small_model
from src.routing.selective_router import route_for_reprocessing
from src.uncertainty.uncertainty_estimator import (
    compute_token_uncertainty,
    identify_uncertain_tokens,
)


def _train_pipeline_models(config):
    """Train the small and large models used by the adaptive pipeline."""

    texts, labels = generate_dataset()
    train_texts, train_labels, test_texts, test_labels = split_dataset(texts, labels)

    small_model, small_vectorizer = train_small_model(
        train_texts,
        train_labels,
        config,
    )
    large_model, large_vectorizer = train_large_model(
        train_texts,
        train_labels,
        config,
    )

    return (
        small_model,
        small_vectorizer,
        large_model,
        large_vectorizer,
        test_texts,
        test_labels,
    )


def _run_sentence_inference(
    sentence: str,
    config,
    small_model,
    small_vectorizer,
    large_model,
    large_vectorizer,
) -> dict[str, object]:
    """Run the adaptive inference flow for a single sentence."""

    small_model_cost_per_token = 1
    large_model_cost_per_token = 3
    num_tokens = len(sentence.split())

    # Stage 1: score the full sentence with the small model.
    prob_small = predict_proba_small(
        small_model,
        small_vectorizer,
        [sentence],
    )[0][1]

    # Stage 2: find token positions that materially affect the prediction.
    token_uncertainty_scores = compute_token_uncertainty(
        small_model,
        small_vectorizer,
        sentence,
        config,
    )
    uncertain_indices = identify_uncertain_tokens(
        small_model,
        small_vectorizer,
        sentence,
        config,
    )

    # Stage 3: keep only uncertain tokens for selective reprocessing.
    routing_info = route_for_reprocessing(sentence, uncertain_indices)
    if not uncertain_indices:
        reduced_sentence = sentence
        prob_large = prob_small
        num_uncertain_tokens = 0
        used_large_model = False
    else:
        reduced_sentence = routing_info["reduced_sentence"]

        # Selective computation: deeper model invoked only for uncertain inputs.
        prob_large = predict_proba_large(
            large_model,
            large_vectorizer,
            [reduced_sentence],
        )[0][1]
        num_uncertain_tokens = len(reduced_sentence.split())
        used_large_model = True

    # The large model is assumed to be more computationally expensive than the
    # small model, so each token routed to it carries a higher cost weight.
    full_compute_cost = large_model_cost_per_token * num_tokens
    selective_compute_cost = (
        small_model_cost_per_token * num_tokens
    ) + (
        large_model_cost_per_token * num_uncertain_tokens
    )

    if full_compute_cost > 0:
        compute_reduction = (
            full_compute_cost - selective_compute_cost
        ) / full_compute_cost
    else:
        compute_reduction = 0.0

    compute_reduction = max(0.0, min(1.0, compute_reduction))

    # Stage 4: fuse the two probabilities into one final decision.
    final_result = fuse_predictions(prob_small, prob_large, config)

    return {
        "sentence": sentence,
        "token_uncertainty_scores": token_uncertainty_scores,
        "uncertain_tokens": routing_info["uncertain_tokens"],
        "reduced_sentence": reduced_sentence,
        "full_compute_cost": full_compute_cost,
        "selective_compute_cost": selective_compute_cost,
        "compute_reduction": float(compute_reduction),
        "prob_small": float(prob_small),
        "prob_large": float(prob_large),
        "final_probability": float(final_result["final_probability"]),
        "final_prediction": int(final_result["final_label"]),
        "used_large_model": used_large_model,
    }


def run_adaptive_inference_pipeline() -> list[dict[str, object]]:
    """Run the full adaptive inference workflow on the synthetic dataset.

    Returns:
        A list of per-sentence result dictionaries containing the original
        sentence, routed uncertain tokens, model probabilities, and the final
        fused prediction.
    """

    config = get_config()
    (
        small_model,
        small_vectorizer,
        large_model,
        large_vectorizer,
        test_texts,
        test_labels,
    ) = _train_pipeline_models(config)

    results: list[dict[str, object]] = []
    for sentence, _ in zip(test_texts, test_labels):
        results.append(
            _run_sentence_inference(
                sentence,
                config,
                small_model,
                small_vectorizer,
                large_model,
                large_vectorizer,
            )
        )

    return results


def run_single_inference(sentence: str) -> dict[str, object]:
    """Run adaptive inference for one user-provided sentence.

    This helper trains the demo models on the synthetic dataset and then runs
    the same selective inference flow used by the batch pipeline on a single
    input sentence. It is useful for interactive experimentation and quick
    manual inspection of the adaptive behavior.
    """

    config = get_config()
    (
        small_model,
        small_vectorizer,
        large_model,
        large_vectorizer,
        _test_texts,
        _test_labels,
    ) = _train_pipeline_models(config)

    return _run_sentence_inference(
        sentence,
        config,
        small_model,
        small_vectorizer,
        large_model,
        large_vectorizer,
    )


class InferencePipeline:
    """Thin wrapper around the module-level pipeline function."""

    def run(self) -> list[dict[str, object]]:
        """Execute the adaptive inference pipeline and return all results."""

        return run_adaptive_inference_pipeline()
