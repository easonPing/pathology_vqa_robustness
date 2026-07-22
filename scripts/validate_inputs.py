#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathology_vqa_exp2.data import build_experiment_cases, load_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen Exp2 inputs.")
    parser.add_argument(
        "--subset", type=Path, default=ROOT / "data/processed/exp2_question_subset.csv"
    )
    parser.add_argument(
        "--paraphrases",
        type=Path,
        default=ROOT / "data/processed/vqa_paraphrases.json",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=ROOT / "data/manifests/slidechat_feature_manifest.csv",
    )
    parser.add_argument(
        "--require-feature-files",
        action="store_true",
        help="Fail unless every referenced SlideChat feature CSV is present.",
    )
    args = parser.parse_args()
    records = load_questions(
        args.subset,
        args.paraphrases,
        feature_manifest_path=args.feature_manifest,
        require_features=args.require_feature_files,
    )
    counts = Counter(record.dataset for record in records)
    case_count = len(build_experiment_cases(records, "slidechat"))
    print(f"questions: {len(records)} ({dict(sorted(counts.items()))})")
    feature_refs = sum(record.feature_path is not None for record in records)
    feature_files = sum(
        record.feature_path is not None and record.feature_path.is_file()
        for record in records
    )
    print(f"feature references: {feature_refs}")
    print(f"feature files present: {feature_files}")
    print(f"cases per model: {case_count}")
    print("validation: ok")


if __name__ == "__main__":
    main()
