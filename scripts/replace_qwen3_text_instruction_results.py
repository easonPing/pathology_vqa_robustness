#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathology_vqa_exp2.io import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    read_latest_jsonl,
)
from pathology_vqa_exp2.prompts import (  # noqa: E402
    EXPERT_SYSTEM_PROMPT,
    QA_JSON_ANSWER_INSTRUCTION,
)


CORE_FILES = (
    "cases.csv",
    "predictions.checkpoint.jsonl",
    "predictions.csv",
    "run_metadata.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace only Qwen3 Text instruction-condition results after a "
            "validated instruction-only rerun."
        )
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--rerun-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--expected-per-condition", type=int, default=2392)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def keyed(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if not case_id:
            raise ValueError(f"{label} contains a row without case_id")
        if case_id in result:
            raise ValueError(f"{label} contains duplicate case_id: {case_id}")
        result[case_id] = row
    return result


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def condition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"qa": 0, "instruction": 0}
    for row in rows:
        condition = str(row.get("prompt_condition", ""))
        if condition not in counts:
            raise ValueError(f"Unexpected prompt_condition: {condition!r}")
        counts[condition] += 1
    return counts


def invalid_count(rows: list[dict[str, Any]], condition: str) -> int:
    return sum(
        row.get("prompt_condition") == condition
        and row.get("predicted_option") == "INVALID"
        for row in rows
    )


def correct_count(rows: list[dict[str, Any]], condition: str) -> int:
    return sum(
        row.get("prompt_condition") == condition
        and bool(row.get("correct"))
        for row in rows
    )


def csv_comparable(value: Any) -> str:
    return "" if value is None else str(value)


def main() -> None:
    args = parse_args()
    if args.expected_per_condition <= 0:
        raise ValueError("--expected-per-condition must be positive")

    base_dir = args.base_dir.resolve()
    rerun_dir = args.rerun_dir.resolve()
    archive_dir = args.archive_dir.resolve()
    for filename in CORE_FILES:
        if not (base_dir / filename).is_file():
            raise FileNotFoundError(base_dir / filename)
        if not (rerun_dir / filename).is_file():
            raise FileNotFoundError(rerun_dir / filename)
    if archive_dir.exists():
        raise FileExistsError(f"Archive directory already exists: {archive_dir}")

    base_cases = read_csv_rows(base_dir / "cases.csv")
    rerun_cases = read_csv_rows(rerun_dir / "cases.csv")
    base_case_by_id = keyed(base_cases, label="base cases")
    rerun_case_by_id = keyed(rerun_cases, label="rerun cases")
    expected = args.expected_per_condition
    if condition_counts(base_cases) != {"qa": expected, "instruction": expected}:
        raise ValueError(
            "Base cases do not contain the expected balanced prompt conditions: "
            f"{condition_counts(base_cases)}"
        )
    if condition_counts(rerun_cases) != {"qa": 0, "instruction": expected}:
        raise ValueError(
            "Rerun cases must contain only the complete instruction condition: "
            f"{condition_counts(rerun_cases)}"
        )

    base_instruction_ids = {
        case_id
        for case_id, row in base_case_by_id.items()
        if row["prompt_condition"] == "instruction"
    }
    if set(rerun_case_by_id) != base_instruction_ids:
        missing = sorted(base_instruction_ids - set(rerun_case_by_id))[:5]
        extra = sorted(set(rerun_case_by_id) - base_instruction_ids)[:5]
        raise ValueError(
            "Rerun instruction IDs do not exactly match the base result; "
            f"missing={missing}, extra={extra}"
        )
    for case_id, row in rerun_case_by_id.items():
        if row.get("model_key") != "qwen3_text":
            raise ValueError(f"{case_id}: unexpected model_key {row.get('model_key')!r}")
        if row.get("system_prompt") != EXPERT_SYSTEM_PROMPT:
            raise ValueError(f"{case_id}: instruction system prompt changed")
        prompt = str(row.get("user_prompt", ""))
        if QA_JSON_ANSWER_INSTRUCTION not in prompt or not prompt.endswith("\nAnswer:"):
            raise ValueError(f"{case_id}: updated JSON/Answer prompt is missing")

    base_predictions = read_latest_jsonl(
        base_dir / "predictions.checkpoint.jsonl", "case_id"
    )
    rerun_predictions = read_latest_jsonl(
        rerun_dir / "predictions.checkpoint.jsonl", "case_id"
    )
    if set(base_predictions) != set(base_case_by_id):
        raise ValueError("Base checkpoint IDs do not match base cases")
    if set(rerun_predictions) != set(rerun_case_by_id):
        raise ValueError("Rerun checkpoint IDs do not match rerun cases")
    failures = [
        case_id
        for case_id, row in rerun_predictions.items()
        if row.get("status") != "ok"
    ]
    if failures:
        raise ValueError(f"Rerun contains failed cases: {failures[:5]}")
    for case_id, case in rerun_case_by_id.items():
        prediction = rerun_predictions[case_id]
        for field, expected_value in case.items():
            if csv_comparable(prediction.get(field)) != csv_comparable(
                expected_value
            ):
                raise ValueError(
                    f"{case_id}: rerun prediction field {field!r} does not "
                    f"match cases.csv"
                )

    base_metadata = json.loads(
        (base_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    rerun_metadata = json.loads(
        (rerun_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    base_adapter = base_metadata.get("adapter", {})
    rerun_adapter = rerun_metadata.get("adapter", {})
    for field in (
        "model_key",
        "model_id",
        "revision",
        "dtype",
        "max_new_tokens",
        "thinking",
        "do_sample",
    ):
        if base_adapter.get(field) != rerun_adapter.get(field):
            raise ValueError(
                f"Adapter field {field!r} changed: "
                f"{base_adapter.get(field)!r} != {rerun_adapter.get(field)!r}"
            )

    merged_cases = [
        rerun_case_by_id.get(row["case_id"], row) for row in base_cases
    ]
    merged_predictions = [
        rerun_predictions.get(row["case_id"], base_predictions[row["case_id"]])
        for row in base_cases
    ]
    if condition_counts(merged_predictions) != {
        "qa": expected,
        "instruction": expected,
    }:
        raise AssertionError("Merged prediction condition counts changed")

    archive_dir.mkdir(parents=True, exist_ok=False)
    for filename in CORE_FILES:
        shutil.copy2(base_dir / filename, archive_dir / filename)

    old_rows = [base_predictions[row["case_id"]] for row in base_cases]
    summary = {
        "replaced_at": datetime.now(timezone.utc).isoformat(),
        "prompt_condition": "instruction",
        "replacement_count": expected,
        "qa_preserved_count": expected,
        "base_dir": str(base_dir),
        "rerun_dir": str(rerun_dir),
        "archive_dir": str(archive_dir),
        "old_instruction_invalid": invalid_count(old_rows, "instruction"),
        "new_instruction_invalid": invalid_count(
            merged_predictions, "instruction"
        ),
        "old_instruction_correct": correct_count(old_rows, "instruction"),
        "new_instruction_correct": correct_count(
            merged_predictions, "instruction"
        ),
        "qa_invalid": invalid_count(merged_predictions, "qa"),
        "qa_correct": correct_count(merged_predictions, "qa"),
    }

    atomic_write_csv(base_dir / "cases.csv", merged_cases)
    write_jsonl_atomic(
        base_dir / "predictions.checkpoint.jsonl", merged_predictions
    )
    atomic_write_csv(base_dir / "predictions.csv", merged_predictions)
    merged_metadata = {
        **base_metadata,
        "finished_at": rerun_metadata.get("finished_at"),
        "case_count": expected * 2,
        "completed_count": expected * 2,
        "failure_count": 0,
        "composite_result": True,
        "instruction_replacement": summary,
        "instruction_rerun_metadata": rerun_metadata,
        "predictions": str((base_dir / "predictions.csv").resolve()),
        "checkpoint": str(
            (base_dir / "predictions.checkpoint.jsonl").resolve()
        ),
    }
    atomic_write_json(base_dir / "run_metadata.json", merged_metadata)
    atomic_write_json(base_dir / "instruction_replacement.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
