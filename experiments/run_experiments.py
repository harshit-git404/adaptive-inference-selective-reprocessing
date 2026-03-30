"""Generate experiment statistics for adaptive inference documentation.

This script runs the adaptive inference pipeline on the configured dataset,
prints a compact quantitative summary, and saves per-sample metrics to a CSV
file for later analysis or patent documentation.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for candidate in (PROJECT_ROOT,):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from config.config import get_config
from src.pipeline.inference_pipeline import run_adaptive_inference_pipeline


def _build_results_table(results: list[dict[str, object]]) -> pd.DataFrame:
    """Convert pipeline results into a flat table for analysis and CSV export."""

    rows: list[dict[str, object]] = []
    for result in results:
        sentence = str(result["sentence"])
        uncertain_tokens = list(result["uncertain_tokens"])
        num_tokens = len(sentence.split())
        num_uncertain_tokens = len(uncertain_tokens)

        rows.append(
            {
                "sentence": sentence,
                "num_tokens": num_tokens,
                "num_uncertain_tokens": num_uncertain_tokens,
                "used_large_model": bool(result["used_large_model"]),
                "compute_reduction": float(result["compute_reduction"]),
                "prob_small": float(result["prob_small"]),
                "prob_large": float(result["prob_large"]),
                "final_probability": float(result["final_probability"]),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "sentence",
            "num_tokens",
            "num_uncertain_tokens",
            "used_large_model",
            "compute_reduction",
            "prob_small",
            "prob_large",
            "final_probability",
        ],
    )


def main() -> None:
    """Run the experiment pipeline and save a results CSV."""

    config = get_config()
    results = run_adaptive_inference_pipeline()
    results_table = _build_results_table(results)

    total_samples = len(results_table)
    average_uncertain_tokens = (
        float(results_table["num_uncertain_tokens"].mean())
        if total_samples
        else 0.0
    )
    large_model_usage = (
        float(results_table["used_large_model"].mean()) * 100
        if total_samples
        else 0.0
    )
    average_compute_reduction = (
        float(results_table["compute_reduction"].mean()) * 100
        if total_samples
        else 0.0
    )
    max_compute_reduction = (
        float(results_table["compute_reduction"].max()) * 100
        if total_samples
        else 0.0
    )
    min_compute_reduction = (
        float(results_table["compute_reduction"].min()) * 100
        if total_samples
        else 0.0
    )
    average_token_count = (
        float(results_table["num_tokens"].mean())
        if total_samples
        else 0.0
    )
    average_selective_token_count = (
        float(results_table["num_uncertain_tokens"].mean())
        if total_samples
        else 0.0
    )

    results_dir = PROJECT_ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.csv"
    results_table.to_csv(results_path, index=False)

    print("EXPERIMENT SUMMARY")
    print()
    print(f"Dataset type: {config['DATASET_TYPE']}")
    print(f"Total samples: {total_samples}")
    print(f"Avg uncertain tokens: {average_uncertain_tokens:.1f}")
    print(f"Large model usage: {large_model_usage:.1f}%")
    print(f"Avg compute reduction: {average_compute_reduction:.1f}%")
    print(f"Max compute reduction: {max_compute_reduction:.1f}%")
    print(f"Min compute reduction: {min_compute_reduction:.1f}%")
    print(f"Avg token count: {average_token_count:.1f}")
    print(f"Avg selective token count: {average_selective_token_count:.1f}")
    print()
    print(f"Saved CSV: {results_path}")


if __name__ == "__main__":
    main()
