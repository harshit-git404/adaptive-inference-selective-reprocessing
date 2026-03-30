"""Run a readable demonstration of the adaptive inference pipeline.

This script executes the end-to-end adaptive inference workflow and prints a
small sample of results so the selective reprocessing behavior is easy to
inspect during experimentation.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for candidate in (PROJECT_ROOT,):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from src.pipeline.inference_pipeline import (
    run_adaptive_inference_pipeline,
    run_single_inference,
)


def _print_title() -> None:
    """Print the demo title shown for both execution modes."""

    print("Adaptive Inference Demonstration")
    print("Selective Feature Reprocessing using Uncertainty Detection")
    print()


def _print_result(result: dict[str, object]) -> None:
    """Print one inference result in the demo format."""

    uncertain_tokens = result["uncertain_tokens"]
    token_uncertainty_scores = result.get("token_uncertainty_scores", [])
    final_prediction = "Positive" if result["final_prediction"] == 1 else "Negative"
    used_large_model = "True" if result["used_large_model"] else "False"

    print("-" * 60)
    print("Input Sentence:")
    print(result["sentence"])
    print()
    print("Uncertain Tokens Detected:")
    print(", ".join(uncertain_tokens) if uncertain_tokens else "None")
    print()
    print("Reduced Sentence (processed by large model):")
    print(result["reduced_sentence"])
    print()
    print("Token uncertainty scores:")
    if token_uncertainty_scores:
        for token_info in token_uncertainty_scores:
            print(
                f"{token_info['token']} -> "
                f"{float(token_info['uncertainty_score']):.3f}"
            )
    else:
        print("None")
    print()
    print("Computation cost:")
    print(f"Full compute cost: {result['full_compute_cost']}")
    print(f"Selective compute cost: {result['selective_compute_cost']}")
    print(f"Compute reduction: {result['compute_reduction'] * 100:.1f}%")
    print()
    print(f"Small Model Probability: {result['prob_small']:.3f}")
    print(f"Large Model Probability: {result['prob_large']:.3f}")
    print(f"Final Probability: {result['final_probability']:.3f}")
    print(f"Used Large Model: {used_large_model}")
    print(f"Final Prediction: {final_prediction}")
    print()


def _print_summary(results: list[dict[str, object]]) -> None:
    """Print summary metrics for a collection of inference results."""

    total_samples = len(results)
    large_model_used = sum(1 for result in results if result["used_large_model"])
    large_model_skipped = total_samples - large_model_used
    skip_percentage = (
        (large_model_skipped / total_samples) * 100 if total_samples else 0.0
    )
    average_uncertain_tokens = (
        sum(len(result["uncertain_tokens"]) for result in results) / total_samples
        if total_samples
        else 0.0
    )
    average_compute_reduction = (
        sum(result["compute_reduction"] for result in results) / total_samples
        if total_samples
        else 0.0
    )
    average_probability_difference = (
        sum(
            abs(result["final_probability"] - result["prob_small"])
            for result in results
        )
        / total_samples
        if total_samples
        else 0.0
    )

    print("=" * 60)
    print("Summary")
    print(f"Total sentences processed: {total_samples}")
    print(f"Average compute reduction: {average_compute_reduction * 100:.1f}%")
    print(f"Large model used: {large_model_used}")
    print(f"Large model skipped: {large_model_skipped}")
    print(f"Average number of uncertain tokens: {average_uncertain_tokens:.2f}")
    print(
        "Average probability difference between small and final prediction: "
        f"{average_probability_difference:.3f}"
    )
    print()
    print("Selective computation statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Large model used: {large_model_used}")
    print(f"Large model skipped: {large_model_skipped}")
    print(f"Skip percentage: {skip_percentage:.0f}%")


def _read_custom_sentences() -> list[str]:
    """Collect one or more custom sentences from the user."""

    print("Enter sentences (press ENTER on empty line to finish):")
    sentences: list[str] = []

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break

        if not line:
            if sentences:
                break
            continue

        sentences.append(line)

    return sentences


def main() -> None:
    """Run the adaptive inference demo and print sample outputs."""

    _print_title()

    try:
        choice = input(
            "Choose mode:\n"
            "1 - demo dataset\n"
            "2 - custom input\n"
            "> "
        ).strip()
    except EOFError:
        choice = "1"

    if choice == "2":
        sentences = _read_custom_sentences()
        if not sentences:
            print("No sentences provided. Exiting.")
            return

        print()
        print("Processing...")
        print()

        if len(sentences) == 1:
            results = [run_single_inference(sentences[0])]
        else:
            results = [run_single_inference(sentence) for sentence in sentences]
        results_to_display = results
    else:
        results = run_adaptive_inference_pipeline()
        results_to_display = results[:10]
        if choice not in {"", "1"}:
            print("Unrecognized choice. Running demo dataset instead.")
            print()

    for result in results_to_display:
        _print_result(result)

    _print_summary(results)


if __name__ == "__main__":
    main()
