#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathology_vqa_exp2.metrics import write_metric_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Exp2 perturbation metrics.")
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="One or more completed predictions.csv files",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in args.predictions]
    predictions = pd.concat(frames, ignore_index=True)
    write_metric_outputs(
        predictions,
        args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(f"wrote metrics: {args.output_dir}")


if __name__ == "__main__":
    main()
