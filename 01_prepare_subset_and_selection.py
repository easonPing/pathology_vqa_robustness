# 01_prepare_subset_and_selection.py

from pathlib import Path
import hashlib
import json

import pandas as pd


PROJECT_DIR = Path.cwd()
HDD_ROOT = Path("/Volumes/TOSHIBA EXT/exp2")

SOURCE_SUBSET_PATH = PROJECT_DIR / "exp2_question_subset_with_slides.csv"
SOURCE_PARAPHRASE_PATH = PROJECT_DIR / "vqa_paraphrases.json"

OUTPUT_SUBSET_PATH = PROJECT_DIR / "exp2_question_subset_native_598.csv"
OUTPUT_PARAPHRASE_PATH = PROJECT_DIR / "vqa_paraphrases_native_598.json"

FEATURE_ROOT = (
    HDD_ROOT
    / "features"
    / "unpacked"
    / "Survey_prompt_feature"
)

TCGA_FEATURE_ROOT = FEATURE_ROOT / "TCGA_patch_feat"
BCNB_FEATURE_ROOT = FEATURE_ROOT / "BCNB_patch_feat"

METADATA_DIR = PROJECT_DIR / "patch_metadata"

SELECTION_PATH = (
    METADATA_DIR
    / "native_patch_selection_407slides_final.csv"
)

EXCLUDED_QIDS = {
    "TCGA-2526",
    "TCGA-2527",
}

PATCHES_PER_SLIDE = 30
MASTER_SEED = 20260714

EXPECTED_QUESTIONS = 598
EXPECTED_SLIDES = 407
EXPECTED_PATCHES = 12210


def normalize_slide_id(value: object) -> str:
    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    if value.isdigit():
        return str(int(value))

    return value


def stable_slide_seed(
    master_seed: int,
    dataset: str,
    slide_id: str,
) -> int:
    payload = (
        f"{master_seed}|{dataset}|{slide_id}"
    ).encode("utf-8")

    return int(
        hashlib.sha256(payload).hexdigest()[:8],
        16,
    )


def find_feature_csv(
    dataset: str,
    slide_id: str,
) -> Path:
    root = (
        TCGA_FEATURE_ROOT
        if dataset == "TCGA"
        else BCNB_FEATURE_ROOT
    )

    matches = [
        path
        for path in root.rglob("*.csv")
        if path.stem == slide_id
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one feature CSV for "
            f"{dataset} slide {slide_id}; "
            f"found {len(matches)}: {matches}"
        )

    return matches[0]


def filter_json_by_qid(
    value,
    included_qids: set[str],
):
    if isinstance(value, list):
        filtered_items = []

        for item in value:
            if (
                isinstance(item, dict)
                and "qid" in item
            ):
                if str(item["qid"]) in included_qids:
                    filtered_items.append(item)
            else:
                filtered = filter_json_by_qid(
                    item,
                    included_qids,
                )

                if filtered is not None:
                    filtered_items.append(filtered)

        return filtered_items

    if isinstance(value, dict):
        if "qid" in value:
            return (
                value
                if str(value["qid"]) in included_qids
                else None
            )

        qid_like_keys = [
            key
            for key in value
            if str(key).startswith(
                (
                    "TCGA-",
                    "BCNB-",
                )
            )
        ]

        if (
            qid_like_keys
            and len(qid_like_keys) == len(value)
        ):
            return {
                key: item
                for key, item in value.items()
                if str(key) in included_qids
            }

        filtered_dict = {}

        for key, item in value.items():
            filtered = filter_json_by_qid(
                item,
                included_qids,
            )

            if filtered is not None:
                filtered_dict[key] = filtered

        return filtered_dict

    return value


def collect_qids(value) -> set[str]:
    found = set()

    if isinstance(value, list):
        for item in value:
            found.update(
                collect_qids(item)
            )

    elif isinstance(value, dict):
        if "qid" in value:
            found.add(
                str(value["qid"])
            )

        for key, item in value.items():
            key_string = str(key)

            if key_string.startswith(
                (
                    "TCGA-",
                    "BCNB-",
                )
            ):
                found.add(key_string)

            found.update(
                collect_qids(item)
            )

    return found


def main() -> None:
    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    assert SOURCE_SUBSET_PATH.exists(), (
        SOURCE_SUBSET_PATH
    )

    assert SOURCE_PARAPHRASE_PATH.exists(), (
        SOURCE_PARAPHRASE_PATH
    )

    assert TCGA_FEATURE_ROOT.exists(), (
        TCGA_FEATURE_ROOT
    )

    assert BCNB_FEATURE_ROOT.exists(), (
        BCNB_FEATURE_ROOT
    )

    subset = pd.read_csv(
        SOURCE_SUBSET_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "qid",
        "dataset",
        "slide_id",
    }

    missing_columns = (
        required_columns
        - set(subset.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing subset columns: "
            f"{sorted(missing_columns)}"
        )

    subset["qid"] = (
        subset["qid"]
        .astype(str)
        .str.strip()
    )

    subset["dataset"] = (
        subset["dataset"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    subset["slide_id"] = (
        subset["slide_id"]
        .map(normalize_slide_id)
    )

    subset_native = subset[
        ~subset["qid"].isin(
            EXCLUDED_QIDS
        )
    ].copy()

    subset_native = (
        subset_native
        .reset_index(drop=True)
    )

    if len(subset_native) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS} included "
            f"questions, found {len(subset_native)}"
        )

    if (
        subset_native["qid"].nunique()
        != EXPECTED_QUESTIONS
    ):
        raise ValueError(
            "Included qids are not unique"
        )

    unique_slides = (
        subset_native[
            [
                "dataset",
                "slide_id",
            ]
        ]
        .drop_duplicates()
    )

    if len(unique_slides) != EXPECTED_SLIDES:
        raise ValueError(
            f"Expected {EXPECTED_SLIDES} unique slides, "
            f"found {len(unique_slides)}"
        )

    subset_native.to_csv(
        OUTPUT_SUBSET_PATH,
        index=False,
    )

    with open(
        SOURCE_PARAPHRASE_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        paraphrase_data = json.load(file)

    included_qids = set(
        subset_native["qid"]
    )

    filtered_paraphrases = (
        filter_json_by_qid(
            paraphrase_data,
            included_qids,
        )
    )

    filtered_qids = collect_qids(
        filtered_paraphrases
    )

    if filtered_qids:
        missing_paraphrases = (
            included_qids
            - filtered_qids
        )

        extra_paraphrases = (
            filtered_qids
            - included_qids
        )

        if missing_paraphrases:
            raise ValueError(
                "Missing paraphrases for qids: "
                f"{sorted(missing_paraphrases)[:20]}"
            )

        if extra_paraphrases:
            raise ValueError(
                "Filtered paraphrase file still contains "
                "excluded qids: "
                f"{sorted(extra_paraphrases)[:20]}"
            )

    with open(
        OUTPUT_PARAPHRASE_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            filtered_paraphrases,
            file,
            indent=2,
            ensure_ascii=False,
        )

    selection_records = []

    sorted_slides = unique_slides.sort_values(
        [
            "dataset",
            "slide_id",
        ]
    )

    for row in sorted_slides.itertuples(
        index=False
    ):
        dataset = row.dataset
        slide_id = row.slide_id

        feature_csv = find_feature_csv(
            dataset,
            slide_id,
        )

        patch_pool = pd.read_csv(
            feature_csv,
            usecols=["patch_name"],
            dtype=str,
            keep_default_na=False,
        )["patch_name"]

        patch_pool = (
            patch_pool
            .astype(str)
            .str.strip()
        )

        patch_pool = (
            patch_pool[
                patch_pool.ne("")
            ]
            .reset_index(drop=True)
        )

        if patch_pool.duplicated().any():
            duplicates = (
                patch_pool[
                    patch_pool.duplicated()
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Duplicate patch names in "
                f"{feature_csv}: "
                f"{duplicates[:10]}"
            )

        if len(patch_pool) < PATCHES_PER_SLIDE:
            raise ValueError(
                f"Slide {dataset}/{slide_id} has only "
                f"{len(patch_pool)} eligible patches"
            )

        slide_seed = stable_slide_seed(
            MASTER_SEED,
            dataset,
            slide_id,
        )

        selected = patch_pool.sample(
            n=PATCHES_PER_SLIDE,
            replace=False,
            random_state=slide_seed,
        ).reset_index(drop=True)

        for patch_index, patch_name in enumerate(
            selected
        ):
            selection_records.append(
                {
                    "dataset": dataset,
                    "slide_id": slide_id,
                    "patch_index": patch_index,
                    "patch_name": patch_name,
                    "eligible_patch_count": len(
                        patch_pool
                    ),
                    "master_seed": MASTER_SEED,
                    "slide_seed": slide_seed,
                    "feature_csv": str(
                        feature_csv
                    ),
                }
            )

    selection = pd.DataFrame(
        selection_records
    )

    per_slide_counts = (
        selection
        .groupby(
            [
                "dataset",
                "slide_id",
            ]
        )
        .size()
    )

    if len(selection) != EXPECTED_PATCHES:
        raise ValueError(
            f"Expected {EXPECTED_PATCHES} selected "
            f"patches, found {len(selection)}"
        )

    if len(per_slide_counts) != EXPECTED_SLIDES:
        raise ValueError(
            f"Expected {EXPECTED_SLIDES} selected "
            f"slides, found {len(per_slide_counts)}"
        )

    if not (
        per_slide_counts
        == PATCHES_PER_SLIDE
    ).all():
        raise ValueError(
            "At least one slide does not have "
            "exactly 30 patches"
        )

    selection.to_csv(
        SELECTION_PATH,
        index=False,
    )

    print(
        f"Saved: {OUTPUT_SUBSET_PATH}"
    )

    print(
        f"Saved: {OUTPUT_PARAPHRASE_PATH}"
    )

    print(
        f"Saved: {SELECTION_PATH}"
    )

    print(
        f"Questions: {len(subset_native)}"
    )

    print(
        f"Slides: {len(per_slide_counts)}"
    )

    print(
        f"Selected patches: {len(selection)}"
    )

    print(
        selection["dataset"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()
