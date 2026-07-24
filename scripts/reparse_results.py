#!/usr/bin/env python3
"""Reparse saved model outputs without modifying inference checkpoints.

The script auto-detects direct (text/SlideChat) and native-patch result
directories.  It always writes to a distinct, initially empty output
directory so the frozen raw JSONL checkpoint remains an audit trail.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathology_vqa_exp2.answer_parser import INVALID_OPTION, parse_answer
from pathology_vqa_exp2.io import atomic_write_csv, atomic_write_json, read_latest_jsonl
from pathology_vqa_exp2.records import ExperimentCase, OPTION_LETTERS
from pathology_vqa_exp2.runner import aggregate_native_patch_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reparse raw outputs into a separate directory without changing "
            "the original checkpoint."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--only-invalid",
        action="store_true",
        help=(
            "Reparse only rows whose saved predicted_option is INVALID; "
            "preserve all existing A-D predictions exactly."
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_image_paths(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError(f"Invalid image_paths value: {value!r}")
    return tuple(parsed)


def _experiment_case(row: dict[str, str]) -> ExperimentCase:
    return ExperimentCase(
        case_id=row["case_id"],
        model_key=row["model_key"],
        qid=row["qid"],
        dataset=row["dataset"],
        source_id=row["source_id"],
        slide=row["slide"],
        task=row["task"],
        question_variant=row["question_variant"],
        prompt_condition=row["prompt_condition"],
        question=row["question"],
        options={letter: row.get(letter, "") for letter in OPTION_LETTERS},
        gold_option=row["gold_option"],
        system_prompt=row.get("system_prompt") or None,
        user_prompt=row["user_prompt"],
        feature_path=row.get("feature_path") or None,
        image_paths=_parse_image_paths(row.get("image_paths", "")),
    )


def _reparse_row(
    row: dict[str, Any],
    *,
    only_invalid: bool = False,
) -> dict[str, Any]:
    updated = dict(row)
    updated["original_predicted_option"] = row.get(
        "predicted_option", INVALID_OPTION
    )
    updated["original_parse_method"] = row.get("parse_method", "")
    if row.get("status") != "ok":
        return updated
    if (
        only_invalid
        and updated["original_predicted_option"] != INVALID_OPTION
    ):
        return updated

    options = {letter: str(row.get(letter, "")) for letter in OPTION_LETTERS}
    parsed = parse_answer(str(row.get("raw_output", "")), options)
    updated["predicted_option"] = parsed.option
    updated["parse_method"] = parsed.method
    updated["correct"] = parsed.option == row.get("gold_option")
    return updated


def _reparse_checkpoint(
    checkpoint: Path,
    ordered_ids: list[str],
    *,
    only_invalid: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    latest = read_latest_jsonl(checkpoint, "case_id")
    missing = [case_id for case_id in ordered_ids if case_id not in latest]
    unexpected = sorted(set(latest) - set(ordered_ids))
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint/case mismatch: {len(missing)} missing and "
            f"{len(unexpected)} unexpected IDs; examples: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    rows = [
        _reparse_row(latest[case_id], only_invalid=only_invalid)
        for case_id in ordered_ids
    ]
    return rows, {str(row["case_id"]): row for row in rows}


def _prediction_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("predicted_option", INVALID_OPTION)) for row in rows)


def _recovery_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        str(row["predicted_option"])
        for row in rows
        if row.get("original_predicted_option") == INVALID_OPTION
        and row.get("predicted_option") != INVALID_OPTION
    )


def _ensure_safe_output(input_dir: Path, output_dir: Path) -> None:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("--output-dir must differ from --input-dir")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be empty to avoid overwrites: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    _ensure_safe_output(input_dir, output_dir)

    cases_path = input_dir / "cases.csv"
    if not cases_path.is_file():
        raise FileNotFoundError(f"Missing cases file: {cases_path}")
    case_rows = _read_csv(cases_path)
    case_ids = [row["case_id"] for row in case_rows]

    patch_checkpoint = input_dir / "patch_predictions.checkpoint.jsonl"
    native = patch_checkpoint.is_file()
    checkpoint = (
        patch_checkpoint
        if native
        else input_dir / "predictions.checkpoint.jsonl"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "source_checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "native_patch_aggregation": native,
        "only_invalid": args.only_invalid,
        "parser_policy": (
            "explicit final/answer labels before option mentions; "
            "no free-text letter guessing"
        ),
    }

    if not native:
        rows, _ = _reparse_checkpoint(
            checkpoint,
            case_ids,
            only_invalid=args.only_invalid,
        )
        atomic_write_csv(output_dir / "predictions.csv", rows)
        before = Counter(
            str(row.get("original_predicted_option", INVALID_OPTION)) for row in rows
        )
        after = _prediction_counts(rows)
        summary.update(
            {
                "case_count": len(rows),
                "original_prediction_counts": dict(before),
                "reparsed_prediction_counts": dict(after),
                "recovered_from_invalid": dict(_recovery_counts(rows)),
            }
        )
    else:
        patch_cases_path = input_dir / "patch_cases.csv"
        if not patch_cases_path.is_file():
            raise FileNotFoundError(f"Missing patch cases file: {patch_cases_path}")
        patch_case_rows = _read_csv(patch_cases_path)
        patch_ids = [row["case_id"] for row in patch_case_rows]
        patch_rows, patch_by_id = _reparse_checkpoint(
            checkpoint,
            patch_ids,
            only_invalid=args.only_invalid,
        )
        atomic_write_csv(output_dir / "patch_predictions.csv", patch_rows)

        cases = [_experiment_case(row) for row in case_rows]
        model_ids = {str(row.get("model_id", "")) for row in patch_rows}
        if len(model_ids) != 1:
            raise ValueError(f"Expected one model_id, found {sorted(model_ids)}")
        predictions = aggregate_native_patch_predictions(
            cases,
            patch_by_id,
            model_id=next(iter(model_ids)),
        )
        atomic_write_csv(output_dir / "predictions.csv", predictions)

        original_predictions_path = input_dir / "predictions.csv"
        original_predictions = _read_csv(original_predictions_path)
        original_by_id = {
            row["case_id"]: row["predicted_option"]
            for row in original_predictions
        }
        changed_aggregates = sum(
            original_by_id.get(str(row["case_id"])) != row["predicted_option"]
            for row in predictions
        )
        summary.update(
            {
                "patch_call_count": len(patch_rows),
                "original_patch_prediction_counts": dict(
                    Counter(
                        str(row.get("original_predicted_option", INVALID_OPTION))
                        for row in patch_rows
                    )
                ),
                "reparsed_patch_prediction_counts": dict(
                    _prediction_counts(patch_rows)
                ),
                "patches_recovered_from_invalid": dict(
                    _recovery_counts(patch_rows)
                ),
                "slide_case_count": len(predictions),
                "original_slide_invalid_count": sum(
                    value == INVALID_OPTION for value in original_by_id.values()
                ),
                "reparsed_slide_invalid_count": sum(
                    row["predicted_option"] == INVALID_OPTION
                    for row in predictions
                ),
                "changed_slide_aggregate_count": changed_aggregates,
            }
        )

    atomic_write_json(output_dir / "reparse_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
