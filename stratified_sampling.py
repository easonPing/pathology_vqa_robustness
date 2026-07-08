"""
Stratified subset sampling for SlideBench-VQA datasets.

- TCGA: ~400 rows, stratified across the 13 tasks in `Narrow Category`.
- BCNB: ~200 rows, stratified across the 7 tasks in `Task`.
"""

import pandas as pd

RANDOM_STATE = 42  # reproducible draw


def allocate(counts: pd.Series, total: int) -> pd.Series:

    proportions = counts / counts.sum()
    exact = proportions * total
    floor = exact.apply(int)               # integer part
    remainder = total - int(floor.sum())   # rows still to hand out

    order = (exact - floor).sort_values(ascending=False).index
    alloc = floor.copy()
    for stratum in order[:remainder]:
        alloc[stratum] += 1

    while (alloc == 0).any():
        zero = alloc.idxmin()
        donor = alloc.idxmax()
        alloc[zero] += 1
        alloc[donor] -= 1

    return alloc.astype(int)


def stratified_sample(df: pd.DataFrame, task_col: str, total: int) -> pd.DataFrame:
    counts = df[task_col].value_counts()
    alloc = allocate(counts, total)

    parts = []
    for task, n in alloc.items():
        n = min(n, counts[task])  # can't sample more than available
        parts.append(df[df[task_col] == task].sample(n=n, random_state=RANDOM_STATE))

    return pd.concat(parts).sort_index()


def report(name: str, original: pd.DataFrame, sample: pd.DataFrame, task_col: str) -> None:
    print(f"\n=== {name}: {len(sample)} sampled from {len(original)} ===")
    orig_pct = original[task_col].value_counts(normalize=True) * 100
    samp_cnt = sample[task_col].value_counts()
    samp_pct = sample[task_col].value_counts(normalize=True) * 100
    print(f"{'task':<40} {'orig%':>7} {'samp%':>7} {'n':>5}")
    for task in orig_pct.index:
        print(f"{task:<40} {orig_pct[task]:6.1f}% {samp_pct.get(task, 0):6.1f}% "
              f"{samp_cnt.get(task, 0):5d}")


def build_id_list(tcga_sample: pd.DataFrame, bcnb_sample: pd.DataFrame) -> pd.DataFrame:

    tcga = pd.DataFrame({
        "dataset": "TCGA",
        "ID": tcga_sample["ID"].values,
        "task": tcga_sample["Narrow Category"].values,
        "Question": tcga_sample["Question"].values,
    })
    bcnb = pd.DataFrame({
        "dataset": "BCNB",
        "ID": bcnb_sample["ID"].values,
        "task": bcnb_sample["Task"].values,
        "Question": bcnb_sample["Question"].values,
    })
    combined = pd.concat([tcga, bcnb], ignore_index=True)
    combined["qid"] = combined["dataset"] + "-" + combined["ID"].astype(str)
    return combined[["qid", "dataset", "ID", "task", "Question"]]


def main() -> None:
    tcga = pd.read_csv("SlideBench-VQA-TCGA.csv")
    bcnb = pd.read_csv("SlideBench-VQA-BCNB.csv")

    tcga_sample = stratified_sample(tcga, "Narrow Category", total=400)
    bcnb_sample = stratified_sample(bcnb, "Task", total=200)

    report("TCGA", tcga, tcga_sample, "Narrow Category")
    report("BCNB", bcnb, bcnb_sample, "Task")

    tcga_sample.to_csv("SlideBench-VQA-TCGA-sample400.csv", index=False)
    bcnb_sample.to_csv("SlideBench-VQA-BCNB-sample200.csv", index=False)

    id_list = build_id_list(tcga_sample, bcnb_sample)
    id_list.to_csv("exp2_question_subset.csv", index=False)

    print("\nWrote SlideBench-VQA-TCGA-sample400.csv and "
          "SlideBench-VQA-BCNB-sample200.csv")
    print(f"Wrote exp2_question_subset.csv: {len(id_list)} questions "
          f"({(id_list['dataset'] == 'TCGA').sum()} TCGA + "
          f"{(id_list['dataset'] == 'BCNB').sum()} BCNB), "
          f"unique qids: {id_list['qid'].nunique()}")


if __name__ == "__main__":
    main()
