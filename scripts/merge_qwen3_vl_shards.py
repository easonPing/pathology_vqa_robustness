#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathology_vqa_exp2.data import build_experiment_cases, load_questions
from pathology_vqa_exp2.io import atomic_write_csv, atomic_write_json
from pathology_vqa_exp2.runner import (
    DEFAULT_NATIVE_PATCH_COUNT,
    aggregate_native_patch_predictions,
    build_native_patch_cases,
    write_cases,
    write_native_patch_cases,
)


CASES_PER_QUESTION = 8
PATCH_CALLS_PER_QUESTION = CASES_PER_QUESTION * DEFAULT_NATIVE_PATCH_COUNT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge disjoint Qwen3-VL question-range checkpoints."
    )
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--paraphrases", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        required=True,
        metavar=("START", "END", "CHECKPOINT"),
        help="Half-open question range and its patch checkpoint",
    )
    return parser.parse_args()


def read_expected_rows(
    path: Path,
    expected_ids: set[str],
) -> dict[str, dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            case_id = str(row.get("case_id", ""))
            if case_id in expected_ids:
                selected[case_id] = row
                if len(selected) == len(expected_ids):
                    break

    missing = expected_ids - set(selected)
    if missing:
        raise ValueError(
            f"{path} is missing {len(missing)} expected patch rows; "
            f"examples: {sorted(missing)[:5]}"
        )
    failed = [
        case_id
        for case_id, row in selected.items()
        if row.get("status") != "ok"
    ]
    if failed:
        raise ValueError(
            f"{path} contains {len(failed)} non-ok patch rows; "
            f"examples: {failed[:5]}"
        )
    return selected


def atomic_write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    records = load_questions(
        args.subset,
        args.paraphrases,
        image_manifest_path=args.image_manifest,
        require_images=True,
        images_per_question=DEFAULT_NATIVE_PATCH_COUNT,
    )
    cases = build_experiment_cases(records, "qwen3_vl")
    patch_cases = build_native_patch_cases(cases)

    sources: list[tuple[int, int, Path]] = []
    for raw_start, raw_end, raw_path in args.source:
        start = int(raw_start)
        end = int(raw_end)
        path = Path(raw_path).resolve()
        if not 0 <= start < end <= len(records):
            raise ValueError(
                f"Invalid source range [{start}, {end}) for {len(records)} questions"
            )
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        sources.append((start, end, path))

    sources.sort(key=lambda item: item[0])
    cursor = 0
    for start, end, _ in sources:
        if start != cursor:
            raise ValueError(
                f"Source ranges must cover all questions contiguously; "
                f"expected range starting at {cursor}, found {start}"
            )
        cursor = end
    if cursor != len(records):
        raise ValueError(
            f"Source ranges end at {cursor}, but {len(records)} questions are required"
        )

    combined: dict[str, dict[str, object]] = {}
    source_metadata: list[dict[str, object]] = []
    for start, end, path in sources:
        expected_slice = patch_cases[
            start * PATCH_CALLS_PER_QUESTION : end * PATCH_CALLS_PER_QUESTION
        ]
        expected_ids = {patch_case.case_id for patch_case in expected_slice}
        rows = read_expected_rows(path, expected_ids)
        overlap = set(combined) & set(rows)
        if overlap:
            raise ValueError(
                f"Source ranges produced duplicate patch IDs: {sorted(overlap)[:5]}"
            )
        combined.update(rows)
        source_metadata.append(
            {
                "question_start": start,
                "question_end": end,
                "question_count": end - start,
                "patch_row_count": len(rows),
                "checkpoint": str(path),
            }
        )

    expected_all = {patch_case.case_id for patch_case in patch_cases}
    if set(combined) != expected_all:
        missing = expected_all - set(combined)
        extra = set(combined) - expected_all
        raise ValueError(
            f"Merged checkpoint mismatch: missing={len(missing)} extra={len(extra)}"
        )

    model_ids = {str(row.get("model_id", "")) for row in combined.values()}
    if len(model_ids) != 1 or not next(iter(model_ids)):
        raise ValueError(f"Expected one non-empty model_id, found {sorted(model_ids)}")
    model_id = next(iter(model_ids))
    ordered_patch_rows = [combined[patch_case.case_id] for patch_case in patch_cases]
    predictions = aggregate_native_patch_predictions(
        cases,
        combined,
        model_id=model_id,
    )
    if len(predictions) != len(cases):
        raise ValueError(
            f"Expected {len(cases)} aggregated predictions, found {len(predictions)}"
        )
    if any(row.get("status") != "ok" for row in predictions):
        raise ValueError("Merged predictions contain incomplete patch aggregation")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_cases(output_dir / "cases.csv", cases)
    write_native_patch_cases(output_dir / "patch_cases.csv", patch_cases)
    atomic_write_jsonl(
        output_dir / "patch_predictions.checkpoint.jsonl",
        ordered_patch_rows,
    )
    atomic_write_csv(output_dir / "patch_predictions.csv", ordered_patch_rows)
    atomic_write_csv(output_dir / "predictions.csv", predictions)
    atomic_write_json(
        output_dir / "run_metadata.json",
        {
            "merged_at": datetime.now(timezone.utc).isoformat(),
            "model_key": "qwen3_vl",
            "model_id": model_id,
            "question_count": len(records),
            "case_count": len(cases),
            "patch_row_count": len(ordered_patch_rows),
            "patches_per_case": DEFAULT_NATIVE_PATCH_COUNT,
            "sources": source_metadata,
        },
    )
    print(
        f"merged {len(ordered_patch_rows)} patch rows into "
        f"{len(predictions)} predictions: {output_dir}"
    )


if __name__ == "__main__":
    main()
