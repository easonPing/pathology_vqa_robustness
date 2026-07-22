from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .answer_parser import INVALID_OPTION, parse_answer
from .io import append_jsonl, atomic_write_csv, atomic_write_json, read_latest_jsonl
from .models.base import ModelAdapter
from .protocol import NATIVE_PATCH_PROTOCOL
from .records import (
    NATIVE_VLM_MODEL_KEYS,
    OPTION_LETTERS,
    ExperimentCase,
    NativePatchCase,
)


DEFAULT_NATIVE_PATCH_COUNT = NATIVE_PATCH_PROTOCOL.patches_per_slide
NATIVE_AGGREGATION_POLICY = NATIVE_PATCH_PROTOCOL.protocol_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_cases(path: str | Path, cases: list[ExperimentCase]) -> None:
    atomic_write_csv(path, [case.to_dict() for case in cases])


def build_native_patch_cases(
    cases: list[ExperimentCase],
    *,
    expected_patch_count: int = DEFAULT_NATIVE_PATCH_COUNT,
) -> list[NativePatchCase]:
    """Expand slide-level native-VLM cases into ordered single-patch calls."""
    if expected_patch_count <= 0:
        raise ValueError("expected_patch_count must be positive")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Slide-level experiment case IDs must be unique")

    patch_cases: list[NativePatchCase] = []
    for case in cases:
        if case.model_key not in NATIVE_VLM_MODEL_KEYS:
            raise ValueError(
                f"{case.case_id}: {case.model_key!r} is not a native-VLM model key"
            )
        if len(case.image_paths) != expected_patch_count:
            raise ValueError(
                f"{case.case_id}: expected exactly {expected_patch_count} patches, "
                f"found {len(case.image_paths)}"
            )
        if len(set(case.image_paths)) != len(case.image_paths):
            raise ValueError(f"{case.case_id}: native patch paths must be unique")

        for patch_index, image_path in enumerate(case.image_paths):
            patch_cases.append(
                NativePatchCase(
                    case_id=f"{case.case_id}::patch::{patch_index:02d}",
                    slide_case_id=case.case_id,
                    patch_index=patch_index,
                    patch_name=Path(image_path).name,
                    image_path=image_path,
                    base_case=case,
                )
            )
    if len({case.case_id for case in patch_cases}) != len(patch_cases):
        raise ValueError("Generated native patch case IDs are not unique")
    return patch_cases


def write_native_patch_cases(
    path: str | Path,
    patch_cases: list[NativePatchCase],
) -> None:
    atomic_write_csv(path, [case.to_dict() for case in patch_cases])


def _validate_resume(
    cases: list[ExperimentCase],
    latest: dict[str, dict[str, Any]],
    adapter_metadata: dict[str, Any],
    previous_metadata: dict[str, Any] | None,
) -> None:
    previous_adapter = (previous_metadata or {}).get("adapter", {})
    for field in ("model_key", "model_id", "revision"):
        previous_value = previous_adapter.get(field)
        current_value = adapter_metadata.get(field)
        if previous_value and current_value and previous_value != current_value:
            raise ValueError(
                f"Cannot resume with a different {field}: "
                f"{previous_value!r} != {current_value!r}"
            )

    for case in cases:
        existing = latest.get(case.case_id)
        if existing is None:
            continue
        expected = case.to_dict()
        expected["model_id"] = adapter_metadata["model_id"]
        for field, expected_value in expected.items():
            if existing.get(field) != expected_value:
                raise ValueError(
                    f"Cannot resume {case.case_id}: {field} changed from "
                    f"{existing.get(field)!r} to {expected_value!r}"
                )


def _validate_native_resume(
    patch_cases: list[NativePatchCase],
    latest: dict[str, dict[str, Any]],
    adapter_metadata: dict[str, Any],
    previous_metadata: dict[str, Any] | None,
    aggregation: dict[str, Any],
) -> None:
    expected_case_ids = {patch_case.case_id for patch_case in patch_cases}
    unexpected_case_ids = set(latest) - expected_case_ids
    if unexpected_case_ids:
        raise ValueError(
            "Cannot resume: patch checkpoint contains cases outside the current "
            f"run: {sorted(unexpected_case_ids)[:5]}"
        )
    _validate_resume(
        [patch_case.model_case() for patch_case in patch_cases],
        latest,
        adapter_metadata,
        previous_metadata,
    )
    for patch_case in patch_cases:
        existing = latest.get(patch_case.case_id)
        if existing is None:
            continue
        for field in ("slide_case_id", "patch_index", "patch_name", "image_path"):
            expected = patch_case.to_dict()[field]
            if existing.get(field) != expected:
                raise ValueError(
                    f"Cannot resume {patch_case.case_id}: {field} changed from "
                    f"{existing.get(field)!r} to {expected!r}"
                )

    previous_aggregation = (previous_metadata or {}).get("aggregation")
    if previous_aggregation and previous_aggregation != aggregation:
        raise ValueError(
            "Cannot resume native-VLM inference with a changed aggregation "
            f"configuration: {previous_aggregation!r} != {aggregation!r}"
        )


def aggregate_native_patch_predictions(
    cases: list[ExperimentCase],
    patch_rows: dict[str, dict[str, Any]],
    *,
    model_id: str,
) -> list[dict[str, Any]]:
    """Aggregate ordered single-patch predictions into slide-level predictions.

    Each parsed response votes for A-D or INVALID. A valid option wins only when
    it is the unique maximum across all five categories. INVALID winning, any
    maximum tie, or zero valid votes produces an INVALID aggregate prediction.
    """
    patch_cases = build_native_patch_cases(
        cases,
        expected_patch_count=DEFAULT_NATIVE_PATCH_COUNT,
    )
    by_slide_case: dict[str, list[NativePatchCase]] = {}
    for patch_case in patch_cases:
        by_slide_case.setdefault(patch_case.slide_case_id, []).append(patch_case)

    aggregated: list[dict[str, Any]] = []
    for case in cases:
        expected = by_slide_case[case.case_id]
        rows = [patch_rows.get(patch_case.case_id) for patch_case in expected]
        missing_ids = [
            patch_case.case_id
            for patch_case, row in zip(expected, rows)
            if row is None
        ]
        available_rows = [row for row in rows if row is not None]
        failures = [row for row in available_rows if row.get("status") != "ok"]
        valid_options = {
            letter
            for letter in OPTION_LETTERS
            if case.options.get(letter, "").strip()
        }
        votes = [
            str(row.get("predicted_option", INVALID_OPTION))
            for row in available_rows
        ]
        normalized_votes = [
            vote if vote in valid_options else INVALID_OPTION for vote in votes
        ]
        valid_votes = [vote for vote in normalized_votes if vote in valid_options]
        invalid_vote_count = len(votes) - len(valid_votes)
        counts = Counter(normalized_votes)
        option_counts = {
            letter: int(counts.get(letter, 0)) for letter in OPTION_LETTERS
        }

        predicted_option = INVALID_OPTION
        parse_method = "majority_vote_no_valid_votes"
        vote_status = "no_valid"
        tied = False
        if not missing_ids and not failures:
            if valid_votes:
                maximum = max(counts.values())
                winners = {
                    option for option, count in counts.items() if count == maximum
                }
                tied = len(winners) > 1
                if len(winners) > 1:
                    parse_method = "majority_vote_tie_invalid"
                    vote_status = "tie"
                elif INVALID_OPTION in winners:
                    parse_method = "majority_vote_invalid_winner"
                    vote_status = "invalid_winner"
                else:
                    predicted_option = next(iter(winners))
                    parse_method = "majority_vote_unique_maximum"
                    vote_status = "unique_valid_winner"
        else:
            parse_method = "patch_inference_incomplete"
            vote_status = "incomplete"

        status = "ok" if not missing_ids and not failures else "error"
        errors: list[str] = []
        if missing_ids:
            errors.append(
                "missing patch results: " + ", ".join(missing_ids[:5])
            )
        if failures:
            errors.append(
                f"{len(failures)} patch calls failed: "
                + "; ".join(str(row.get("error", "")) for row in failures[:3])
            )
        vote_summary = {
            "counts": {**option_counts, INVALID_OPTION: invalid_vote_count},
            "valid": len(valid_votes),
            "total": len(available_rows),
        }
        row = {
            **case.to_dict(),
            "model_id": model_id,
            "raw_output": json.dumps(vote_summary, sort_keys=True),
            "predicted_option": predicted_option,
            "parse_method": parse_method,
            "correct": predicted_option == case.gold_option,
            "status": status,
            "error": " | ".join(errors),
            "latency_seconds": round(
                sum(float(item.get("latency_seconds", 0.0)) for item in available_rows),
                6,
            ),
            "recorded_at": _utc_now(),
            "aggregation_method": "single_patch_majority_vote",
            "aggregation_policy": NATIVE_AGGREGATION_POLICY,
            "expected_patch_count": DEFAULT_NATIVE_PATCH_COUNT,
            "patch_call_count": len(available_rows),
            "valid_vote_count": len(valid_votes),
            "invalid_vote_count": invalid_vote_count,
            "vote_tied": tied,
            "vote_status": vote_status,
            "tie_policy": "invalid",
            "invalid_vote_policy": "fifth_vote_category_no_retry",
            **{f"vote_{letter}": option_counts[letter] for letter in OPTION_LETTERS},
            "vote_INVALID": invalid_vote_count,
        }
        aggregated.append(row)
    return aggregated


def run_cases(
    adapter: ModelAdapter,
    cases: list[ExperimentCase],
    output_dir: str | Path,
    *,
    resume: bool = True,
    fail_fast: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.csv"
    checkpoint_path = output_dir / "predictions.checkpoint.jsonl"
    predictions_path = output_dir / "predictions.csv"
    metadata_path = output_dir / "run_metadata.json"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(
            f"Checkpoint already exists; use resume or a new output directory: "
            f"{checkpoint_path}"
        )
    latest = read_latest_jsonl(checkpoint_path, "case_id") if resume else {}
    previous_metadata = None
    if metadata_path.exists():
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    adapter_metadata = adapter.metadata()
    _validate_resume(cases, latest, adapter_metadata, previous_metadata)
    write_cases(cases_path, cases)
    completed = {
        case_id for case_id, row in latest.items() if row.get("status") == "ok"
    }

    metadata = {
        "started_at": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "case_count": len(cases),
        "resume": resume,
        "already_completed": len(completed),
        "adapter": adapter_metadata,
    }
    atomic_write_json(metadata_path, metadata)

    for index, case in enumerate(cases, start=1):
        if case.case_id in completed:
            print(f"[{index}/{len(cases)}] {case.case_id}: already complete")
            continue

        started = time.perf_counter()
        row = {
            **case.to_dict(),
            "model_id": adapter.model_id,
            "raw_output": "",
            "predicted_option": INVALID_OPTION,
            "parse_method": "not_run",
            "correct": False,
            "status": "error",
            "error": "",
            "error_traceback": "",
            "latency_seconds": 0.0,
            "recorded_at": "",
        }
        try:
            raw_output = adapter.generate(case)
            parsed = parse_answer(raw_output, case.options)
            row.update(
                {
                    "raw_output": raw_output,
                    "predicted_option": parsed.option,
                    "parse_method": parsed.method,
                    "correct": parsed.option == case.gold_option,
                    "status": "ok",
                }
            )
        except Exception as exc:  # Preserve the failed case before propagating.
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["error_traceback"] = traceback.format_exc()
        finally:
            row["latency_seconds"] = round(time.perf_counter() - started, 6)
            row["recorded_at"] = _utc_now()
            append_jsonl(checkpoint_path, row)
            latest[case.case_id] = row

        print(
            f"[{index}/{len(cases)}] {case.case_id}: {row['status']} "
            f"({row['predicted_option']})"
        )
        if row["status"] != "ok" and fail_fast:
            raise RuntimeError(row["error"])

    ordered_rows = [latest[case.case_id] for case in cases if case.case_id in latest]
    atomic_write_csv(predictions_path, ordered_rows)
    failures = [row for row in ordered_rows if row.get("status") != "ok"]
    metadata.update(
        {
            "finished_at": _utc_now(),
            "completed_count": len(ordered_rows) - len(failures),
            "failure_count": len(failures),
            "predictions": str(predictions_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
        }
    )
    atomic_write_json(metadata_path, metadata)
    if failures:
        raise RuntimeError(
            f"{len(failures)} cases failed; inspect {predictions_path} and rerun "
            "with resume enabled"
        )
    return metadata


def run_native_patch_cases(
    adapter: ModelAdapter,
    cases: list[ExperimentCase],
    output_dir: str | Path,
    *,
    resume: bool = True,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Run native VLMs once per frozen patch and aggregate by majority vote."""
    if adapter.model_key not in NATIVE_VLM_MODEL_KEYS:
        raise ValueError(
            f"run_native_patch_cases requires a native-VLM adapter, found "
            f"{adapter.model_key!r}"
        )
    if any(case.model_key != adapter.model_key for case in cases):
        raise ValueError("Every experiment case must match the adapter model_key")
    patch_cases = build_native_patch_cases(
        cases,
        expected_patch_count=DEFAULT_NATIVE_PATCH_COUNT,
    )
    aggregation = {
        "method": "single_patch_majority_vote",
        "aggregation_policy": NATIVE_AGGREGATION_POLICY,
        "patches_per_case": DEFAULT_NATIVE_PATCH_COUNT,
        "winner_rule": "unique_maximum_across_A_B_C_D_INVALID",
        "invalid_responses": "fifth_vote_category_without_retry",
        "tie_or_no_valid_vote": INVALID_OPTION,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.csv"
    patch_cases_path = output_dir / "patch_cases.csv"
    checkpoint_path = output_dir / "patch_predictions.checkpoint.jsonl"
    patch_predictions_path = output_dir / "patch_predictions.csv"
    predictions_path = output_dir / "predictions.csv"
    metadata_path = output_dir / "run_metadata.json"
    legacy_checkpoint = output_dir / "predictions.checkpoint.jsonl"
    if legacy_checkpoint.exists() and not checkpoint_path.exists():
        raise ValueError(
            "This output directory contains a legacy multi-image checkpoint. "
            "Use a new output directory for the single-patch protocol: "
            f"{legacy_checkpoint}"
        )
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(
            "Patch checkpoint already exists; use resume or a new output "
            f"directory: {checkpoint_path}"
        )

    latest = read_latest_jsonl(checkpoint_path, "case_id") if resume else {}
    previous_metadata = None
    if metadata_path.exists():
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    adapter_metadata = adapter.metadata()
    _validate_native_resume(
        patch_cases,
        latest,
        adapter_metadata,
        previous_metadata,
        aggregation,
    )
    write_cases(cases_path, cases)
    write_native_patch_cases(patch_cases_path, patch_cases)
    completed = {
        case_id for case_id, row in latest.items() if row.get("status") == "ok"
    }

    metadata = {
        "started_at": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "case_count": len(cases),
        "patch_call_count": len(patch_cases),
        "resume": resume,
        "already_completed_patch_calls": len(completed),
        "adapter": adapter_metadata,
        "aggregation": aggregation,
        "aggregation_policy": NATIVE_AGGREGATION_POLICY,
        "patches_per_case": DEFAULT_NATIVE_PATCH_COUNT,
    }
    atomic_write_json(metadata_path, metadata)

    for index, patch_case in enumerate(patch_cases, start=1):
        if patch_case.case_id in completed:
            print(
                f"[{index}/{len(patch_cases)}] {patch_case.case_id}: "
                "already complete"
            )
            continue

        model_case = patch_case.model_case()
        started = time.perf_counter()
        row = {
            **patch_case.to_dict(),
            "model_id": adapter.model_id,
            "raw_output": "",
            "predicted_option": INVALID_OPTION,
            "parse_method": "not_run",
            "correct": False,
            "status": "error",
            "error": "",
            "error_traceback": "",
            "latency_seconds": 0.0,
            "recorded_at": "",
        }
        try:
            raw_output = adapter.generate(model_case)
            parsed = parse_answer(raw_output, model_case.options)
            row.update(
                {
                    "raw_output": raw_output,
                    "predicted_option": parsed.option,
                    "parse_method": parsed.method,
                    "correct": parsed.option == model_case.gold_option,
                    "status": "ok",
                }
            )
        except Exception as exc:  # Preserve the failed patch before propagating.
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["error_traceback"] = traceback.format_exc()
        finally:
            row["latency_seconds"] = round(time.perf_counter() - started, 6)
            row["recorded_at"] = _utc_now()
            append_jsonl(checkpoint_path, row)
            latest[patch_case.case_id] = row

        print(
            f"[{index}/{len(patch_cases)}] {patch_case.case_id}: "
            f"{row['status']} ({row['predicted_option']})"
        )
        if row["status"] != "ok" and fail_fast:
            raise RuntimeError(row["error"])

    ordered_patch_rows = [
        latest[patch_case.case_id]
        for patch_case in patch_cases
        if patch_case.case_id in latest
    ]
    atomic_write_csv(patch_predictions_path, ordered_patch_rows)
    predictions = aggregate_native_patch_predictions(
        cases,
        latest,
        model_id=adapter.model_id,
    )
    atomic_write_csv(predictions_path, predictions)

    patch_failures = [
        row for row in ordered_patch_rows if row.get("status") != "ok"
    ]
    metadata.update(
        {
            "finished_at": _utc_now(),
            "completed_patch_call_count": len(ordered_patch_rows)
            - len(patch_failures),
            "patch_failure_count": len(patch_failures),
            "completed_slide_case_count": sum(
                row.get("status") == "ok" for row in predictions
            ),
            "slide_case_failure_count": sum(
                row.get("status") != "ok" for row in predictions
            ),
            "predictions": str(predictions_path.resolve()),
            "patch_predictions": str(patch_predictions_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
        }
    )
    atomic_write_json(metadata_path, metadata)
    if patch_failures:
        raise RuntimeError(
            f"{len(patch_failures)} patch calls failed; inspect "
            f"{patch_predictions_path} and rerun with resume enabled"
        )
    return metadata
