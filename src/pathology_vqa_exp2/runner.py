from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .answer_parser import INVALID_OPTION, parse_answer
from .io import append_jsonl, atomic_write_csv, atomic_write_json, read_latest_jsonl
from .models.base import ModelAdapter
from .records import ExperimentCase


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_cases(path: str | Path, cases: list[ExperimentCase]) -> None:
    atomic_write_csv(path, [case.to_dict() for case in cases])


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
