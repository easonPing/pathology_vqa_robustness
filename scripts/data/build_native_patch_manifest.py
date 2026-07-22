#!/usr/bin/env python3
"""Join the frozen questions to the verified 30-patch slide report.

The extraction report is slide-level (407 slides x 30 patches).  Native VLM
inference consumes a qid-level manifest so questions that share a slide also
share the exact same ordered patch paths and checksums.  By default this script
re-opens and hashes every image before writing the canonical manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBSET = REPO_ROOT / "data/processed/exp2_question_subset.csv"
DEFAULT_REPORT = (
    REPO_ROOT / "data/manifests/native_patch_extraction_report.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "data/manifests/native_patch_manifest.csv"
DEFAULT_PROTOCOL = REPO_ROOT / "configs/native_patch_protocol.json"

EXPECTED_QUESTIONS = 598
EXPECTED_SLIDES = 407
EXPECTED_QUESTIONS_BY_DATASET = {"TCGA": 398, "BCNB": 200}
EXPECTED_SLIDES_BY_DATASET = {"TCGA": 353, "BCNB": 54}
EXCLUDED_QIDS = {"TCGA-2526", "TCGA-2527"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the verified qid-level native-patch manifest."
    )
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--patch-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--skip-file-verification",
        action="store_true",
        help=(
            "Only validate report metadata. Production manifests should not "
            "use this option."
        ),
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Path must be inside the repository: {resolved}") from error
    return resolved


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def normalize_slide(value: object) -> str:
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return str(int(value)) if value.isdigit() else value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_patch_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    count = payload.get("patches_per_slide")
    if count != 30:
        raise ValueError(
            f"This experiment requires patches_per_slide=30; found {count!r}"
        )
    return int(count)


def load_subset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"qid", "dataset", "Slide"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Subset is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["qid"] = frame["qid"].str.strip()
    frame["dataset"] = frame["dataset"].str.strip().str.upper()
    frame["slide"] = frame["Slide"].map(normalize_slide)
    if len(frame) != EXPECTED_QUESTIONS:
        raise ValueError(f"Expected 598 frozen questions, found {len(frame)}")
    if frame["qid"].duplicated().any():
        raise ValueError("Frozen subset contains duplicate qids")
    if EXCLUDED_QIDS & set(frame["qid"]):
        raise ValueError("Frozen subset contains a PhD-approved excluded qid")
    question_counts = frame["dataset"].value_counts().to_dict()
    if question_counts != EXPECTED_QUESTIONS_BY_DATASET:
        raise ValueError(
            f"Frozen question counts changed: {question_counts}"
        )
    slides = frame[["dataset", "slide"]].drop_duplicates()
    if len(slides) != EXPECTED_SLIDES:
        raise ValueError(f"Expected 407 frozen slides, found {len(slides)}")
    slide_counts = slides["dataset"].value_counts().to_dict()
    if slide_counts != EXPECTED_SLIDES_BY_DATASET:
        raise ValueError(f"Frozen slide counts changed: {slide_counts}")
    return frame[["qid", "dataset", "slide"]]


def load_report(path: Path, *, patches_per_slide: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    aliases = {
        "slide_id": "slide",
        "patch_sha256": "sha256",
        "patch_path": "image_path",
    }
    for old, new in aliases.items():
        if new not in frame.columns and old in frame.columns:
            frame = frame.rename(columns={old: new})
    required = {
        "dataset",
        "slide",
        "patch_index",
        "patch_name",
        "image_path",
        "sha256",
        "status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Patch report is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["dataset"] = frame["dataset"].str.strip().str.upper()
    frame["slide"] = frame["slide"].map(normalize_slide)
    frame["patch_name"] = frame["patch_name"].str.strip()
    frame["image_path"] = frame["image_path"].str.strip()
    frame["sha256"] = frame["sha256"].str.strip().str.lower()
    try:
        frame["patch_index"] = frame["patch_index"].map(int)
    except (TypeError, ValueError) as error:
        raise ValueError("Every patch_index must be an integer") from error

    if set(frame["status"]) != {"found"}:
        counts = frame["status"].value_counts(dropna=False).to_dict()
        raise ValueError(f"Patch report is not fully successful: {counts}")
    if not frame["sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Every found patch must have a valid SHA256")
    if frame.duplicated(["dataset", "slide", "patch_index"]).any():
        raise ValueError("Patch report contains duplicate slide/patch_index rows")
    if frame.duplicated(["dataset", "slide", "patch_name"]).any():
        raise ValueError("Patch report contains duplicate slide/patch_name rows")

    groups = frame.groupby(["dataset", "slide"], sort=True)
    if groups.ngroups != EXPECTED_SLIDES:
        raise ValueError(f"Expected 407 report slides, found {groups.ngroups}")
    expected_indices = list(range(patches_per_slide))
    bad = [
        f"{dataset}/{slide}"
        for (dataset, slide), group in groups
        if sorted(group["patch_index"].tolist()) != expected_indices
    ]
    if bad:
        raise ValueError(f"Slides without patch indexes 0..29: {bad[:20]}")
    if len(frame) != EXPECTED_SLIDES * patches_per_slide:
        raise ValueError(f"Expected 12,210 patch rows, found {len(frame)}")

    portable_paths: list[str] = []
    for row in frame.to_dict("records"):
        raw_path = Path(row["image_path"])
        image_path = raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path
        image_path = image_path.resolve()
        try:
            portable = repo_relative(image_path)
        except ValueError as error:
            raise ValueError(
                f"Patch path must be repository-relative: {raw_path}"
            ) from error
        if image_path.name != row["patch_name"]:
            raise ValueError(
                f"Patch basename mismatch: {image_path.name!r} != "
                f"{row['patch_name']!r}"
            )
        portable_paths.append(portable)
    frame["image_path"] = portable_paths
    return frame[
        [
            "dataset",
            "slide",
            "patch_index",
            "patch_name",
            "image_path",
            "sha256",
            "status",
        ]
    ].sort_values(["dataset", "slide", "patch_index"])


def verify_files(frame: pd.DataFrame) -> None:
    for index, row in enumerate(frame.to_dict("records"), start=1):
        path = (REPO_ROOT / row["image_path"]).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing or empty patch: {path}")
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != (224, 224) or image.mode != "RGB":
                    raise ValueError(
                        f"Patch must be 224x224 RGB, found "
                        f"{image.size}/{image.mode}: {path}"
                    )
        except OSError as error:
            raise ValueError(f"Unreadable patch image: {path}") from error
        observed = sha256_file(path)
        if observed != row["sha256"]:
            raise ValueError(
                f"SHA256 mismatch for {path}: {observed} != {row['sha256']}"
            )
        if index % 1000 == 0:
            print(f"Verified {index}/{len(frame)} slide-level patches")


def write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    subset_path = resolve_repo_path(args.subset)
    report_path = resolve_repo_path(args.patch_report)
    output_path = resolve_repo_path(args.output)
    protocol_path = resolve_repo_path(args.protocol_config)
    patches_per_slide = load_patch_count(protocol_path)
    subset = load_subset(subset_path)
    report = load_report(report_path, patches_per_slide=patches_per_slide)

    subset_slides = set(map(tuple, subset[["dataset", "slide"]].to_numpy()))
    report_slides = set(map(tuple, report[["dataset", "slide"]].to_numpy()))
    if subset_slides != report_slides:
        raise ValueError(
            "Patch-report slides do not exactly match the frozen subset: "
            f"missing={sorted(subset_slides - report_slides)[:10]}, "
            f"extra={sorted(report_slides - subset_slides)[:10]}"
        )
    if not args.skip_file_verification:
        verify_files(report)

    manifest = subset.merge(
        report,
        on=["dataset", "slide"],
        how="left",
        validate="many_to_many",
        sort=False,
    )
    manifest = manifest[
        [
            "qid",
            "dataset",
            "slide",
            "patch_index",
            "patch_name",
            "image_path",
            "sha256",
            "status",
        ]
    ].sort_values(["qid", "patch_index"], kind="stable")
    expected_rows = EXPECTED_QUESTIONS * patches_per_slide
    if len(manifest) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} qid-level rows, found {len(manifest):,}"
        )
    if not (
        manifest.groupby("qid", sort=False).size() == patches_per_slide
    ).all():
        raise ValueError("Every frozen qid must have exactly 30 patches")

    write_atomic(manifest, output_path)
    print(f"Saved: {repo_relative(output_path)}")
    print(f"Questions: {subset['qid'].nunique()}")
    print(f"Slides: {len(subset_slides)}")
    print(f"Manifest rows: {len(manifest):,}")
    print(f"Files verified: {not args.skip_file_verification}")


if __name__ == "__main__":
    main()
