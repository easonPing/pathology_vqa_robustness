from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .images import NativeImageSet, load_native_image_manifest
from .prompts import render_prompt
from .records import (
    OPTION_LETTERS,
    PROMPT_CONDITIONS,
    QUESTION_VARIANTS,
    ExperimentCase,
    QuestionRecord,
)


REQUIRED_SUBSET_COLUMNS = {
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
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _load_feature_paths(
    manifest_path: Path,
    repo_root: Path,
) -> dict[str, dict[str, object]]:
    manifest = _read_csv(manifest_path)
    required = {"qid", "dataset", "slide", "status", "feat_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Feature manifest is missing columns: {sorted(missing)}")
    if manifest["qid"].duplicated().any():
        duplicates = manifest.loc[manifest["qid"].duplicated(), "qid"].tolist()
        raise ValueError(f"Duplicate qids in feature manifest: {duplicates[:5]}")

    feature_paths: dict[str, dict[str, object]] = {}
    for row in manifest.to_dict("records"):
        if row["status"] != "found" or not row["feat_path"]:
            continue
        path = Path(row["feat_path"])
        if not path.is_absolute():
            path = repo_root / path
        feature_paths[row["qid"]] = {
            "dataset": row.get("dataset", ""),
            "slide": row.get("slide", ""),
            "path": path.resolve(),
        }
    return feature_paths


def load_questions(
    subset_path: str | Path,
    paraphrases_path: str | Path,
    *,
    feature_manifest_path: str | Path | None = None,
    require_features: bool = False,
    image_manifest_path: str | Path | None = None,
    require_images: bool = False,
    images_per_question: int | None = None,
    qids: list[str] | tuple[str, ...] | None = None,
) -> list[QuestionRecord]:
    subset_path = Path(subset_path).resolve()
    paraphrases_path = Path(paraphrases_path).resolve()
    subset = _read_csv(subset_path)
    missing = REQUIRED_SUBSET_COLUMNS - set(subset.columns)
    if missing:
        raise ValueError(f"Question subset is missing columns: {sorted(missing)}")
    if subset["qid"].duplicated().any():
        duplicates = subset.loc[subset["qid"].duplicated(), "qid"].tolist()
        raise ValueError(f"Duplicate qids in question subset: {duplicates[:5]}")

    payload = json.loads(paraphrases_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("Paraphrase JSON must contain an items list")
    paraphrase_items = {str(item.get("qid")): item for item in payload["items"]}
    if len(paraphrase_items) != len(payload["items"]):
        raise ValueError("Duplicate or missing qid in paraphrase JSON")

    repo_root = subset_path.parent.parent.parent
    feature_paths: dict[str, dict[str, object]] = {}
    if feature_manifest_path is not None:
        feature_paths = _load_feature_paths(
            Path(feature_manifest_path).resolve(), repo_root
        )

    if require_images and image_manifest_path is None:
        raise ValueError("Native-VLM inference requires an image manifest")
    if images_per_question is not None and images_per_question <= 0:
        raise ValueError("images_per_question must be positive")
    image_sets: dict[str, NativeImageSet] = {}
    if image_manifest_path is not None:
        image_sets = load_native_image_manifest(
            image_manifest_path,
            repo_root=repo_root,
            require_files=False,
        )
        extra_image_qids = set(image_sets) - set(subset["qid"])
        if extra_image_qids:
            raise ValueError(
                "Image manifest contains qids outside the frozen subset: "
                f"{sorted(extra_image_qids)[:5]}"
            )

    selected = set(qids) if qids else None
    records: list[QuestionRecord] = []
    for row in subset.to_dict("records"):
        qid = row["qid"]
        if selected is not None and qid not in selected:
            continue
        item = paraphrase_items.get(qid)
        if item is None:
            raise ValueError(f"Missing paraphrases for {qid}")

        options = {letter: row[letter].strip() for letter in OPTION_LETTERS}
        json_options = {
            letter: str(item.get("options", {}).get(letter, "")).strip()
            for letter in OPTION_LETTERS
        }
        consistency_fields = {
            "dataset": (row["dataset"], str(item.get("dataset", ""))),
            "ID": (row["ID"], str(item.get("ID", ""))),
            "Slide": (row["Slide"], str(item.get("Slide", ""))),
            "Task": (row["Task"], str(item.get("Task", ""))),
            "Question": (row["Question"], str(item.get("Question", ""))),
            "Answer": (row["Answer"], str(item.get("Answer", ""))),
        }
        for name, (csv_value, json_value) in consistency_fields.items():
            if csv_value != json_value:
                raise ValueError(
                    f"{qid}: {name} differs between subset CSV and paraphrase JSON"
                )
        if options != json_options:
            raise ValueError(
                f"{qid}: options differ between subset CSV and paraphrase JSON"
            )

        answer = row["Answer"].strip().upper()
        if answer not in OPTION_LETTERS or not options.get(answer):
            raise ValueError(f"{qid}: invalid gold answer {answer!r}")

        paraphrase_entries = item.get("paraphrases")
        if not isinstance(paraphrase_entries, list) or len(paraphrase_entries) != 3:
            raise ValueError(f"{qid}: expected exactly three paraphrases")
        paraphrases: dict[str, str] = {}
        for index, entry in enumerate(paraphrase_entries, start=1):
            expected_id = f"{qid}-p{index}"
            if entry.get("paraphrase_id") != expected_id:
                raise ValueError(
                    f"{qid}: expected paraphrase_id {expected_id}, "
                    f"found {entry.get('paraphrase_id')!r}"
                )
            text = str(entry.get("Question", "")).strip()
            if not text:
                raise ValueError(f"{qid}: paraphrase p{index} is empty")
            paraphrases[f"p{index}"] = text

        feature_entry = feature_paths.get(qid)
        feature_path = feature_entry["path"] if feature_entry else None
        if feature_entry is not None:
            if feature_entry["dataset"] != row["dataset"]:
                raise ValueError(f"{qid}: dataset differs from feature manifest")
            if feature_entry["slide"] != row["Slide"]:
                raise ValueError(f"{qid}: slide differs from feature manifest")
        if require_features:
            if feature_path is None:
                raise ValueError(f"{qid}: no found feature in manifest")
            if not feature_path.is_file():
                raise FileNotFoundError(f"{qid}: feature does not exist: {feature_path}")

        image_entry = image_sets.get(qid)
        image_paths = image_entry.paths if image_entry else ()
        if image_entry is not None:
            if image_entry.dataset != row["dataset"]:
                raise ValueError(f"{qid}: dataset differs from image manifest")
            if image_entry.slide != row["Slide"]:
                raise ValueError(f"{qid}: slide differs from image manifest")
        if require_images:
            if image_entry is None:
                raise ValueError(f"{qid}: no entries in image manifest")
            if image_entry.missing_patch_names:
                names = list(image_entry.missing_patch_names[:5])
                raise ValueError(f"{qid}: image manifest has missing patches: {names}")
            if not image_paths:
                raise ValueError(f"{qid}: image manifest contains no found patches")
            for image_path in image_paths:
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"{qid}: image does not exist: {image_path}"
                    )
        if images_per_question is not None and image_entry is not None:
            if len(image_paths) != images_per_question:
                raise ValueError(
                    f"{qid}: expected {images_per_question} found patches, "
                    f"found {len(image_paths)}"
                )

        records.append(
            QuestionRecord(
                qid=qid,
                dataset=row["dataset"],
                source_id=row["ID"],
                slide=row["Slide"],
                task=row["Task"],
                question=row["Question"],
                options=options,
                answer=answer,
                paraphrases=paraphrases,
                feature_path=feature_path,
                image_paths=image_paths,
            )
        )

    if selected is not None:
        found = {record.qid for record in records}
        missing_qids = selected - found
        if missing_qids:
            raise ValueError(f"Requested qids not found: {sorted(missing_qids)}")
    if not records:
        raise ValueError("No questions were selected")
    return records


def build_experiment_cases(
    records: list[QuestionRecord],
    model_key: str,
) -> list[ExperimentCase]:
    cases: list[ExperimentCase] = []
    for record in records:
        variants = {"canonical": record.question, **record.paraphrases}
        if tuple(variants) != QUESTION_VARIANTS:
            raise ValueError(f"{record.qid}: unexpected question variants")
        for variant, question in variants.items():
            for condition in PROMPT_CONDITIONS:
                system_prompt, user_prompt = render_prompt(
                    model_key, question, record.options, condition
                )
                cases.append(
                    ExperimentCase(
                        case_id=f"{record.qid}::{variant}::{condition}",
                        model_key=model_key,
                        qid=record.qid,
                        dataset=record.dataset,
                        source_id=record.source_id,
                        slide=record.slide,
                        task=record.task,
                        question_variant=variant,
                        prompt_condition=condition,
                        question=question,
                        options=record.options,
                        gold_option=record.answer,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        feature_path=(
                            str(record.feature_path) if record.feature_path else None
                        ),
                        image_paths=tuple(str(path) for path in record.image_paths),
                    )
                )
    return cases
