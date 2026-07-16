from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_IMAGE_MANIFEST_COLUMNS = {
    "qid",
    "dataset",
    "slide",
    "patch_index",
    "patch_name",
    "image_path",
    "sha256",
    "status",
}
VALID_IMAGE_STATUSES = {"found", "missing"}


@dataclass(frozen=True)
class NativeImageSet:
    dataset: str
    slide: str
    paths: tuple[Path, ...]
    missing_patch_names: tuple[str, ...]


def load_native_image_manifest(
    manifest_path: str | Path,
    *,
    repo_root: str | Path,
    require_files: bool = False,
) -> dict[str, NativeImageSet]:
    """Load the frozen, ordered native-VLM patch selection for each question."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    missing_columns = REQUIRED_IMAGE_MANIFEST_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Image manifest is missing columns: {sorted(missing_columns)}"
        )
    if frame.empty:
        raise ValueError("Image manifest is empty")

    invalid_statuses = set(frame["status"]) - VALID_IMAGE_STATUSES
    if invalid_statuses:
        raise ValueError(
            f"Image manifest has invalid statuses: {sorted(invalid_statuses)}"
        )
    if frame.duplicated(["qid", "patch_index"]).any():
        duplicates = frame.loc[
            frame.duplicated(["qid", "patch_index"], keep=False),
            ["qid", "patch_index"],
        ].head()
        raise ValueError(
            "Duplicate qid/patch_index rows in image manifest:\n"
            f"{duplicates.to_string(index=False)}"
        )
    if frame.duplicated(["qid", "patch_name"]).any():
        duplicates = frame.loc[
            frame.duplicated(["qid", "patch_name"], keep=False),
            ["qid", "patch_name"],
        ].head()
        raise ValueError(
            "Duplicate qid/patch_name rows in image manifest:\n"
            f"{duplicates.to_string(index=False)}"
        )

    try:
        frame["patch_index_int"] = frame["patch_index"].map(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("Every image manifest patch_index must be an integer") from exc
    if (frame["patch_index_int"] < 0).any():
        raise ValueError("Image manifest patch_index values must be non-negative")

    image_sets: dict[str, NativeImageSet] = {}
    slide_signatures: dict[tuple[str, str], tuple[tuple[str, ...], ...]] = {}
    for qid, group in frame.groupby("qid", sort=False):
        if not qid:
            raise ValueError("Image manifest contains an empty qid")
        datasets = set(group["dataset"])
        slides = set(group["slide"])
        if len(datasets) != 1 or len(slides) != 1:
            raise ValueError(f"{qid}: image manifest has inconsistent metadata")

        group = group.sort_values("patch_index_int")
        expected_indices = list(range(len(group)))
        observed_indices = group["patch_index_int"].tolist()
        if observed_indices != expected_indices:
            raise ValueError(
                f"{qid}: patch_index must be contiguous from zero; "
                f"found {observed_indices[:10]}"
            )

        signature = tuple(
            (
                row["patch_name"].strip(),
                row["image_path"].strip(),
                row["sha256"].strip().lower(),
                row["status"],
            )
            for row in group.to_dict("records")
        )
        slide_key = (next(iter(datasets)), next(iter(slides)))
        previous_signature = slide_signatures.setdefault(slide_key, signature)
        if signature != previous_signature:
            raise ValueError(
                f"{qid}: qids sharing {slide_key[0]}/{slide_key[1]} must use "
                "identical ordered patches"
            )

        paths: list[Path] = []
        missing_patch_names: list[str] = []
        for row in group.to_dict("records"):
            patch_name = row["patch_name"].strip()
            if not patch_name:
                raise ValueError(f"{qid}: patch_name is empty")
            if row["status"] == "missing":
                if row["image_path"].strip() or row["sha256"].strip():
                    raise ValueError(
                        f"{qid}/{patch_name}: missing row must have empty "
                        "image_path and sha256"
                    )
                missing_patch_names.append(patch_name)
                continue

            checksum = row["sha256"].strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
                raise ValueError(
                    f"{qid}/{patch_name}: sha256 must be 64 hexadecimal characters"
                )
            raw_path = row["image_path"].strip()
            if not raw_path:
                raise ValueError(f"{qid}/{patch_name}: found row has no image_path")
            path = Path(raw_path)
            if not path.is_absolute():
                path = repo_root / path
            path = path.resolve()
            if path.name != patch_name:
                raise ValueError(
                    f"{qid}: image_path basename {path.name!r} does not match "
                    f"patch_name {patch_name!r}"
                )
            if require_files and not path.is_file():
                raise FileNotFoundError(f"{qid}: image does not exist: {path}")
            paths.append(path)

        image_sets[qid] = NativeImageSet(
            dataset=next(iter(datasets)),
            slide=next(iter(slides)),
            paths=tuple(paths),
            missing_patch_names=tuple(missing_patch_names),
        )
    return image_sets
