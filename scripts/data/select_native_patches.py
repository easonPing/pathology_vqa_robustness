#!/usr/bin/env python3
"""Freeze the 30 raw-image patches used by the native-VLM baseline.

SlideChat's public protocol samples 30 patches per WSI.  This script samples
from the ``patch_name`` column of the delivered SlideChat feature CSVs so that
the native VLMs and SlideChat refer to the same patch pool.  It deliberately
does not create, filter, or rewrite the frozen 598-question subset or its
paraphrases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SUBSET_PATH = REPO_ROOT / "data/processed/exp2_question_subset.csv"
DEFAULT_FEATURE_ROOT = REPO_ROOT / "data/features/slidechat"
DEFAULT_SELECTION_PATH = REPO_ROOT / "data/manifests/native_patch_selection.csv"
DEFAULT_PROTOCOL_PATH = REPO_ROOT / "configs/native_patch_protocol.json"

EXPECTED_QUESTIONS_BY_DATASET = {"TCGA": 398, "BCNB": 200}
EXPECTED_SLIDES_BY_DATASET = {"TCGA": 353, "BCNB": 54}
EXCLUDED_QIDS = {"TCGA-2526", "TCGA-2527"}
SUPPORTED_DATASETS = frozenset(EXPECTED_QUESTIONS_BY_DATASET)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically select 30 SlideChat feature patches per WSI "
            "for the native-VLM experiment."
        )
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=DEFAULT_SUBSET_PATH,
        help="Frozen 598-question subset CSV.",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_FEATURE_ROOT,
        help=(
            "Root containing TCGA_patch_feat/ and BCNB_patch_feat/. Nested "
            "cancer-type directories are supported."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SELECTION_PATH,
        help="Output slide-level patch-selection manifest.",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help="Single source of truth for the frozen native-patch protocol.",
    )
    parser.add_argument(
        "--patches-per-slide",
        type=int,
        default=None,
        help="Explicit override; must agree with the protocol config by default.",
    )
    parser.add_argument(
        "--allow-protocol-override",
        action="store_true",
        help=(
            "Acknowledge a CLI/config mismatch. The production study remains "
            "locked to 30 patches."
        ),
    )
    parser.add_argument("--master-seed", type=int, default=20260714)
    parser.add_argument("--expected-questions", type=int, default=598)
    parser.add_argument("--expected-slides", type=int, default=407)
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    """Resolve CLI paths relative to the repository, never the launch cwd."""
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def repo_relative(path: Path) -> str:
    """Return a portable repository-relative POSIX path."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Path must be inside the repository: {path}") from error


def load_patches_per_slide(
    protocol_path: Path,
    *,
    cli_value: int | None,
    allow_override: bool,
) -> int:
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing native-patch protocol: {protocol_path}")
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if not isinstance(protocol, dict):
        raise ValueError("Native-patch protocol must be a JSON object")
    configured = protocol.get("patches_per_slide")
    if not isinstance(configured, int) or isinstance(configured, bool):
        raise ValueError("Protocol patches_per_slide must be an integer")
    if cli_value is not None and cli_value != configured and not allow_override:
        raise ValueError(
            f"CLI patches_per_slide={cli_value} disagrees with protocol "
            f"value {configured}; pass --allow-protocol-override only for an "
            "intentional, audited override"
        )
    selected = cli_value if cli_value is not None else configured
    if selected != 30:
        raise ValueError(
            "This production experiment is locked to the SlideChat baseline "
            "of exactly 30 patches per slide"
        )
    return selected


def normalize_slide_id(value: object) -> str:
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    if value.isdigit():
        return str(int(value))
    return value


def stable_slide_seed(master_seed: int, dataset: str, slide: str) -> int:
    payload = f"{master_seed}|{dataset}|{slide}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def validate_patch_name(patch_name: str, *, feature_csv: Path) -> None:
    candidate = Path(patch_name)
    if (
        not patch_name
        or candidate.is_absolute()
        or candidate.name != patch_name
        or patch_name in {".", ".."}
    ):
        raise ValueError(
            f"Unsafe or non-basename patch_name {patch_name!r} in {feature_csv}"
        )


def build_feature_index(root: Path, dataset: str) -> dict[str, Path]:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")

    dataset_root = root / f"{dataset}_patch_feat"
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Missing {dataset} feature directory: {dataset_root}"
        )

    index: dict[str, Path] = {}
    for path in sorted(dataset_root.rglob("*.csv")):
        slide = normalize_slide_id(path.stem)
        if slide in index:
            raise ValueError(
                f"Duplicate feature CSVs for {dataset}/{slide}: "
                f"{index[slide]} and {path}"
            )
        index[slide] = path.resolve()
    return index


def load_and_validate_subset(
    path: Path,
    *,
    expected_questions: int,
    expected_slides: int,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen question subset: {path}")

    subset = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"qid", "dataset", "Slide"}
    missing = required - set(subset.columns)
    if missing:
        raise ValueError(f"Missing subset columns: {sorted(missing)}")

    subset = subset.copy()
    subset["qid"] = subset["qid"].str.strip()
    subset["dataset"] = subset["dataset"].str.strip().str.upper()
    subset["slide"] = subset["Slide"].map(normalize_slide_id)

    unsupported = set(subset["dataset"]) - SUPPORTED_DATASETS
    if unsupported:
        raise ValueError(f"Unsupported dataset values: {sorted(unsupported)}")
    if (subset["qid"] == "").any() or (subset["slide"] == "").any():
        raise ValueError("qid and Slide values must be non-empty")
    if subset["qid"].duplicated().any():
        duplicates = subset.loc[subset["qid"].duplicated(), "qid"].tolist()
        raise ValueError(f"Duplicate qids in frozen subset: {duplicates[:20]}")

    accidentally_included = EXCLUDED_QIDS & set(subset["qid"])
    if accidentally_included:
        raise ValueError(
            "The frozen subset contains the two PhD-approved exclusions: "
            f"{sorted(accidentally_included)}"
        )
    if len(subset) != expected_questions:
        raise ValueError(
            f"Expected {expected_questions} frozen questions, found {len(subset)}"
        )

    question_counts = subset["dataset"].value_counts().to_dict()
    if question_counts != EXPECTED_QUESTIONS_BY_DATASET:
        raise ValueError(
            "Frozen question counts changed; expected "
            f"{EXPECTED_QUESTIONS_BY_DATASET}, found {question_counts}"
        )

    slides = subset[["dataset", "slide"]].drop_duplicates()
    if len(slides) != expected_slides:
        raise ValueError(
            f"Expected {expected_slides} unique slides, found {len(slides)}"
        )
    slide_counts = slides["dataset"].value_counts().to_dict()
    if slide_counts != EXPECTED_SLIDES_BY_DATASET:
        raise ValueError(
            "Frozen slide counts changed; expected "
            f"{EXPECTED_SLIDES_BY_DATASET}, found {slide_counts}"
        )
    return slides.sort_values(["dataset", "slide"]).reset_index(drop=True)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    subset_path = resolve_repo_path(args.subset)
    feature_root = resolve_repo_path(args.feature_root)
    output_path = resolve_repo_path(args.output)
    protocol_path = resolve_repo_path(args.protocol_config)
    for path in (subset_path, feature_root, output_path, protocol_path):
        repo_relative(path)
    patches_per_slide = load_patches_per_slide(
        protocol_path,
        cli_value=args.patches_per_slide,
        allow_override=args.allow_protocol_override,
    )

    if args.expected_questions != 598 or args.expected_slides != 407:
        raise ValueError(
            "This script only prepares the frozen 598-question/407-slide study"
        )

    slides = load_and_validate_subset(
        subset_path,
        expected_questions=args.expected_questions,
        expected_slides=args.expected_slides,
    )
    feature_indexes = {
        dataset: build_feature_index(feature_root, dataset)
        for dataset in sorted(SUPPORTED_DATASETS)
    }

    records: list[dict[str, object]] = []
    for row in slides.itertuples(index=False):
        dataset = row.dataset
        slide = row.slide
        feature_csv = feature_indexes[dataset].get(slide)
        if feature_csv is None:
            raise FileNotFoundError(f"Missing feature CSV for {dataset}/{slide}")

        patch_pool = pd.read_csv(
            feature_csv,
            usecols=["patch_name"],
            dtype=str,
            keep_default_na=False,
        )["patch_name"].str.strip()
        patch_pool = patch_pool[patch_pool.ne("")].reset_index(drop=True)
        for patch_name in patch_pool:
            validate_patch_name(patch_name, feature_csv=feature_csv)
        if patch_pool.duplicated().any():
            duplicates = patch_pool[patch_pool.duplicated()].unique().tolist()
            raise ValueError(
                f"Duplicate patch names in {feature_csv}: {duplicates[:10]}"
            )
        if len(patch_pool) < patches_per_slide:
            raise ValueError(
                f"{dataset}/{slide} has only {len(patch_pool)} eligible patches"
            )
        patch_pool_sha256 = hashlib.sha256(
            "\n".join(patch_pool.tolist()).encode("utf-8")
        ).hexdigest()

        slide_seed = stable_slide_seed(args.master_seed, dataset, slide)
        # Keep Harry's seeded pandas sampling algorithm so an existing frozen
        # handoff can be reproduced exactly. Sampling is without replacement.
        selected = patch_pool.sample(
            n=patches_per_slide,
            replace=False,
            random_state=slide_seed,
        ).reset_index(drop=True)

        for patch_index, patch_name in enumerate(selected):
            records.append(
                {
                    "dataset": dataset,
                    "slide": slide,
                    "patch_index": patch_index,
                    "patch_name": patch_name,
                    "eligible_patch_count": len(patch_pool),
                    "ordered_patch_pool_sha256": patch_pool_sha256,
                    "master_seed": args.master_seed,
                    "slide_seed": slide_seed,
                    "selection_algorithm": "pandas_series_sample_without_replacement",
                    "pandas_version": pd.__version__,
                    "feature_path": repo_relative(feature_csv),
                }
            )

    selection = pd.DataFrame.from_records(records)
    expected_patches = args.expected_slides * patches_per_slide
    if len(selection) != expected_patches:
        raise ValueError(
            f"Expected {expected_patches} selected patches, found {len(selection)}"
        )
    per_slide = selection.groupby(["dataset", "slide"], sort=True).size()
    if len(per_slide) != args.expected_slides or not (
        per_slide == patches_per_slide
    ).all():
        raise ValueError("Every frozen slide must have exactly 30 selected patches")

    write_csv_atomic(selection, output_path)
    print(f"Saved: {repo_relative(output_path)}")
    print(f"Protocol: {repo_relative(protocol_path)}")
    print(f"Questions (unchanged): {args.expected_questions}")
    print(f"Slides: {len(per_slide)}")
    print(f"Selected patches: {len(selection)}")
    print(selection["dataset"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
