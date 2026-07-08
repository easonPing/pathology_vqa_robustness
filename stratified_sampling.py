
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


OUTPUT_COLUMNS = [
    "qid",
    "dataset",
    "ID",
    "Slide",
    "Task",
    "Question",
    "A",
    "B",
    "C",
    "D",
    "Answer",
]

DATASETS = ("TCGA", "BCNB")
OPTION_COLUMNS = ("A", "B", "C", "D")


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def task_column(dataset: str) -> str:
    if dataset == "TCGA":
        return "Narrow Category"
    if dataset == "BCNB":
        return "Task"
    raise ValueError(f"Unknown dataset: {dataset}")


def load_benchmark_by_id(tcga_csv: Path, bcnb_csv: Path) -> Dict[str, Dict[str, dict]]:
    paths = {"TCGA": tcga_csv, "BCNB": bcnb_csv}
    loaded: Dict[str, Dict[str, dict]] = {}
    for dataset, path in paths.items():
        rows = read_csv(path)
        by_id: Dict[str, dict] = {}
        duplicates = []
        for row in rows:
            row_id = row["ID"]
            if row_id in by_id:
                duplicates.append(row_id)
            by_id[row_id] = row
        if duplicates:
            sample = ", ".join(duplicates[:5])
            raise ValueError(f"{dataset} has duplicate IDs in {path}: {sample}")
        loaded[dataset] = by_id
    return loaded


def normalize_benchmark_row(dataset: str, source: Mapping[str, str], qid: str | None = None) -> dict:
    answer = source.get("Answer", "").strip()
    if answer not in OPTION_COLUMNS:
        raise ValueError(f"{dataset}-{source.get('ID')} has invalid Answer={answer!r}")
    if not source.get(answer, "").strip():
        raise ValueError(
            f"{dataset}-{source.get('ID')} Answer={answer!r} points to an empty option"
        )

    normalized = {
        "qid": qid or f"{dataset}-{source['ID']}",
        "dataset": dataset,
        "ID": source["ID"],
        "Slide": source["Slide"],
        "Task": source[task_column(dataset)],
        "Question": source["Question"],
        "A": source.get("A", ""),
        "B": source.get("B", ""),
        "C": source.get("C", ""),
        "D": source.get("D", ""),
        "Answer": answer,
    }
    return normalized


def largest_remainder_quotas(counts: Mapping[str, int], target: int) -> Dict[str, int]:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Cannot compute quotas from empty counts")
    raw = {task: count / total * target for task, count in counts.items()}
    quotas = {task: int(value) for task, value in raw.items()}
    remainder = target - sum(quotas.values())
    order = sorted(
        raw,
        key=lambda task: (raw[task] - quotas[task], counts[task], task),
        reverse=True,
    )
    for task in order[:remainder]:
        quotas[task] += 1
    return quotas


def expected_quotas(benchmark_rows: Iterable[Mapping[str, str]], dataset: str, target: int) -> Dict[str, int]:
    column = task_column(dataset)
    return largest_remainder_quotas(Counter(row[column] for row in benchmark_rows), target)


def validate_distribution(
    rows: Sequence[Mapping[str, str]],
    benchmark_by_id: Mapping[str, Mapping[str, dict]],
    tcga_n: int,
    bcnb_n: int,
) -> None:
    targets = {"TCGA": tcga_n, "BCNB": bcnb_n}
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    for dataset in DATASETS:
        actual_n = len(by_dataset[dataset])
        expected_n = targets[dataset]
        if actual_n != expected_n:
            raise ValueError(f"{dataset} subset has {actual_n} rows, expected {expected_n}")

        benchmark_rows = list(benchmark_by_id[dataset].values())
        expected = expected_quotas(benchmark_rows, dataset, expected_n)
        actual = Counter(row["Task"] for row in by_dataset[dataset])
        if actual != expected:
            raise ValueError(
                f"{dataset} task distribution mismatch.\n"
                f"Expected: {dict(sorted(expected.items()))}\n"
                f"Actual:   {dict(sorted(actual.items()))}"
            )


def prepare_fixed_subset(
    subset: Path,
    benchmark_by_id: Mapping[str, Mapping[str, dict]],
    tcga_n: int,
    bcnb_n: int,
) -> List[dict]:
    existing_rows = read_csv(subset)
    required = {"qid", "dataset", "ID", "Question"}
    missing = required - set(existing_rows[0].keys() if existing_rows else [])
    if missing:
        raise ValueError(f"{subset} missing required columns: {sorted(missing)}")

    seen_qids = set()
    prepared = []
    for row in existing_rows:
        dataset = row["dataset"].strip()
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset in subset: {dataset!r}")
        qid = row["qid"].strip()
        if qid in seen_qids:
            raise ValueError(f"Duplicate qid in subset: {qid}")
        seen_qids.add(qid)

        row_id = row["ID"].strip()
        source = benchmark_by_id[dataset].get(row_id)
        if source is None:
            raise ValueError(f"{dataset}-{row_id} not found in benchmark CSV")

        source_task = source[task_column(dataset)]
        subset_task = row.get("Task") or row.get("task")
        if subset_task is not None and subset_task != source_task:
            raise ValueError(
                f"{qid} task mismatch: subset={subset_task!r}, benchmark={source_task!r}"
            )
        if row["Question"] != source["Question"]:
            raise ValueError(f"{qid} question text does not match benchmark CSV")

        prepared.append(normalize_benchmark_row(dataset, source, qid=qid))

    validate_distribution(prepared, benchmark_by_id, tcga_n=tcga_n, bcnb_n=bcnb_n)
    return prepared


def print_summary(rows: Sequence[Mapping[str, str]]) -> None:
    print(f"rows: {len(rows)}")
    print(f"columns: {','.join(OUTPUT_COLUMNS)}")
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        print(f"{dataset}: {len(dataset_rows)}")
        for task, count in Counter(row["Task"] for row in dataset_rows).most_common():
            print(f"  {task}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcga-csv", type=Path, default=Path("data/raw/SlideBench-VQA-TCGA.csv"))
    parser.add_argument("--bcnb-csv", type=Path, default=Path("data/raw/SlideBench-VQA-BCNB.csv"))
    parser.add_argument("--subset", type=Path, default=Path("data/exp2_question_subset.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/exp2_question_subset.csv"))
    parser.add_argument("--tcga-n", type=int, default=400)
    parser.add_argument("--bcnb-n", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_by_id = load_benchmark_by_id(args.tcga_csv, args.bcnb_csv)
    rows = prepare_fixed_subset(
        args.subset,
        benchmark_by_id,
        tcga_n=args.tcga_n,
        bcnb_n=args.bcnb_n,
    )
    write_csv(args.output, rows)
    print_summary(rows)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
