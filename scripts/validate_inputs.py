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
        default=ROOT
        / "data/features/survey_prompt_feature/exp2_feature_manifest.csv",
    )
    args = parser.parse_args()
    records = load_questions(
        args.subset,
        args.paraphrases,
        feature_manifest_path=args.feature_manifest,
        require_features=True,
    )
    counts = Counter(record.dataset for record in records)
    case_count = len(build_experiment_cases(records, "slidechat"))
    print(f"questions: {len(records)} ({dict(sorted(counts.items()))})")
    print(f"features: {sum(record.feature_path is not None for record in records)}")
    print(f"cases per model: {case_count}")
    print("validation: ok")


if __name__ == "__main__":
    main()
