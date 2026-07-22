from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2] / "configs/native_patch_protocol.json"
)


@dataclass(frozen=True)
class NativePatchProtocol:
    protocol_id: str
    patches_per_slide: int
    vote_categories: tuple[str, ...]


def load_native_patch_protocol(
    path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> NativePatchProtocol:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_values = {
        "protocol_id": "five_category_unique_mode_v1",
        "patches_per_slide": 30,
        "inference_unit": "one_generation_per_patch",
        "generation_sampling": False,
        "vote_categories": ["A", "B", "C", "D", "INVALID"],
        "aggregation": "unique_plurality",
        "invalid_policy": "fifth_vote_category_no_retry",
        "tie_policy": "aggregate_prediction_INVALID",
        "retry_invalid_output": False,
        "extra_tie_break_patch": False,
    }
    mismatches = {
        key: {"expected": expected, "found": payload.get(key)}
        for key, expected in required_values.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"Native-patch protocol does not match the frozen study: {mismatches}"
        )
    return NativePatchProtocol(
        protocol_id=payload["protocol_id"],
        patches_per_slide=payload["patches_per_slide"],
        vote_categories=tuple(payload["vote_categories"]),
    )


NATIVE_PATCH_PROTOCOL = load_native_patch_protocol()
