from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .answer_parser import INVALID_OPTION
from .io import atomic_write_csv
from .records import OPTION_LETTERS, PROMPT_CONDITIONS, QUESTION_VARIANTS


PERTURBATION_GROUPS = ("paraphrase", "prompt", "paraphrase+prompt")
PREDICTION_CATEGORIES = (*OPTION_LETTERS, INVALID_OPTION)


def _group_mask(frame: pd.DataFrame, group: str) -> pd.Series:
    canonical = frame["question_variant"] == "canonical"
    qa_prompt = frame["prompt_condition"] == "qa"
    if group == "paraphrase":
        return ~canonical & qa_prompt
    if group == "prompt":
        return canonical & ~qa_prompt
    if group == "paraphrase+prompt":
        return ~canonical & ~qa_prompt
    raise ValueError(f"Unknown perturbation group: {group}")


def validate_predictions(frame: pd.DataFrame) -> None:
    required = {
        "case_id",
        "model_key",
        "qid",
        "dataset",
        "question_variant",
        "prompt_condition",
        "gold_option",
        "predicted_option",
        "status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Predictions are empty")
    duplicate_mask = frame.duplicated(["model_key", "case_id"], keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, ["model_key", "case_id"]].head()
        raise ValueError(f"Duplicate model/case rows:\n{examples.to_string(index=False)}")
    failures = frame[frame["status"] != "ok"]
    if not failures.empty:
        raise ValueError(
            f"Cannot compute metrics: {len(failures)} inference cases failed"
        )
    invalid_categories = set(frame["predicted_option"]) - set(PREDICTION_CATEGORIES)
    if invalid_categories:
        raise ValueError(
            f"Unknown predicted_option values: {sorted(invalid_categories)}"
        )

    expected = {
        (variant, condition)
        for variant in QUESTION_VARIANTS
        for condition in PROMPT_CONDITIONS
    }
    for (model_key, qid), group in frame.groupby(["model_key", "qid"]):
        observed = set(zip(group["question_variant"], group["prompt_condition"]))
        expected_count = len(expected)
        if observed != expected or len(group) != expected_count:
            missing_cases = sorted(expected - observed)
            extra_cases = sorted(observed - expected)
            raise ValueError(
                f"{model_key}/{qid}: expected {expected_count} unique combinations; "
                f"missing={missing_cases}, extra={extra_cases}, rows={len(group)}"
            )


def _entropy(values: list[str]) -> float:
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _pairwise_flip_rate(values: list[str]) -> float:
    pairs = list(itertools.combinations(values, 2))
    if not pairs:
        return 0.0
    return float(np.mean([left != right for left, right in pairs]))


def _bootstrap_intervals(
    canonical_scores: np.ndarray,
    perturbed_scores: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if len(canonical_scores) != len(perturbed_scores) or not len(canonical_scores):
        raise ValueError("bootstrap inputs must be non-empty and aligned by question")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(canonical_scores), size=(replicates, len(canonical_scores))
    )
    canonical_samples = canonical_scores[indices].mean(axis=1)
    perturbed_samples = perturbed_scores[indices].mean(axis=1)
    delta_samples = canonical_samples - perturbed_samples

    def interval(values: np.ndarray) -> tuple[float, float]:
        low, high = np.percentile(values, [2.5, 97.5])
        return float(low), float(high)

    canonical_low, canonical_high = interval(canonical_samples)
    perturbed_low, perturbed_high = interval(perturbed_samples)
    delta_low, delta_high = interval(delta_samples)
    return {
        "canonical_ci_low": canonical_low,
        "canonical_ci_high": canonical_high,
        "perturbed_ci_low": perturbed_low,
        "perturbed_ci_high": perturbed_high,
        "delta_ci_low": delta_low,
        "delta_ci_high": delta_high,
    }


def compute_metrics(
    frame: pd.DataFrame,
    *,
    bootstrap_replicates: int = 10_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_predictions(frame)
    frame = frame.copy()
    frame["correct_derived"] = frame["predicted_option"] == frame["gold_option"]
    metric_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []

    for model_key, model_frame in frame.groupby("model_key", sort=True):
        model_ids = sorted(set(model_frame.get("model_id", pd.Series(dtype=str))))
        model_id = model_ids[0] if len(model_ids) == 1 else ";".join(model_ids)
        for perturbation in PERTURBATION_GROUPS:
            per_question: list[dict[str, Any]] = []
            all_perturbed: list[str] = []
            for qid, question_frame in model_frame.groupby("qid", sort=True):
                anchor = question_frame[
                    (question_frame["question_variant"] == "canonical")
                    & (question_frame["prompt_condition"] == "qa")
                ]
                perturbed = question_frame[_group_mask(question_frame, perturbation)]
                if len(anchor) != 1 or perturbed.empty:
                    raise ValueError(
                        f"{model_key}/{qid}/{perturbation}: invalid anchor or group"
                    )
                anchor_row = anchor.iloc[0]
                values = [anchor_row["predicted_option"], *perturbed["predicted_option"]]
                canonical_correct = float(anchor_row["correct_derived"])
                perturbed_accuracy = float(perturbed["correct_derived"].mean())
                row = {
                    "model_key": model_key,
                    "model_id": model_id,
                    "perturbation": perturbation,
                    "qid": qid,
                    "dataset": anchor_row["dataset"],
                    "canonical_correct": canonical_correct,
                    "perturbed_accuracy": perturbed_accuracy,
                    "accuracy_delta": canonical_correct - perturbed_accuracy,
                    "consistency": float(len(set(values)) == 1),
                    "flip_rate": _pairwise_flip_rate(values),
                    "entropy_bits": _entropy(values),
                    "perturbed_variant_count": len(perturbed),
                }
                per_question.append(row)
                question_rows.append(row)
                all_perturbed.extend(perturbed["predicted_option"].tolist())

            per_question_frame = pd.DataFrame(per_question)
            canonical_scores = per_question_frame["canonical_correct"].to_numpy(float)
            perturbed_scores = per_question_frame["perturbed_accuracy"].to_numpy(float)
            canonical_accuracy = float(canonical_scores.mean())
            perturbed_accuracy = float(perturbed_scores.mean())
            intervals = _bootstrap_intervals(
                canonical_scores,
                perturbed_scores,
                replicates=bootstrap_replicates,
                seed=seed,
            )
            metric_rows.append(
                {
                    "model_key": model_key,
                    "model_id": model_id,
                    "perturbation": perturbation,
                    "question_count": len(per_question_frame),
                    "perturbed_prediction_count": len(all_perturbed),
                    "canonical_accuracy": canonical_accuracy,
                    "perturbed_accuracy": perturbed_accuracy,
                    "accuracy_delta": canonical_accuracy - perturbed_accuracy,
                    **intervals,
                    "consistency": float(per_question_frame["consistency"].mean()),
                    "flip_rate": float(per_question_frame["flip_rate"].mean()),
                    "entropy_bits": float(per_question_frame["entropy_bits"].mean()),
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_seed": seed,
                }
            )
            counts = pd.Series(all_perturbed).value_counts()
            for option in PREDICTION_CATEGORIES:
                count = int(counts.get(option, 0))
                position_rows.append(
                    {
                        "model_key": model_key,
                        "model_id": model_id,
                        "perturbation": perturbation,
                        "selected_option": option,
                        "count": count,
                        "proportion": count / len(all_perturbed),
                    }
                )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(question_rows),
        pd.DataFrame(position_rows),
    )


def _format_estimate(value: float, low: float, high: float) -> str:
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def write_metric_outputs(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 10_000,
    seed: int = 42,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, per_question, positions = compute_metrics(
        predictions,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    atomic_write_csv(
        output_dir / "per_model_perturbation_metrics.csv",
        metrics.to_dict("records"),
    )
    atomic_write_csv(
        output_dir / "per_question_metrics.csv",
        per_question.to_dict("records"),
    )
    atomic_write_csv(
        output_dir / "position_bias.csv",
        positions.to_dict("records"),
    )

    lines = [
        "# Exp2 Table 4 Rows",
        "",
        "Accuracy entries are estimate [question-level bootstrap 95% CI].",
        "",
        "| Model | Perturbation | Canonical accuracy | Perturbed accuracy | Delta | Consistency | Flip-rate | Entropy (bits) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict("records"):
        lines.append(
            "| {model_key} | {perturbation} | {canonical} | {perturbed} | "
            "{delta} | {consistency:.3f} | {flip_rate:.3f} | "
            "{entropy_bits:.3f} |".format(
                **row,
                canonical=_format_estimate(
                    row["canonical_accuracy"],
                    row["canonical_ci_low"],
                    row["canonical_ci_high"],
                ),
                perturbed=_format_estimate(
                    row["perturbed_accuracy"],
                    row["perturbed_ci_low"],
                    row["perturbed_ci_high"],
                ),
                delta=_format_estimate(
                    row["accuracy_delta"],
                    row["delta_ci_low"],
                    row["delta_ci_high"],
                ),
            )
        )
    (output_dir / "exp2_table4_row.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
