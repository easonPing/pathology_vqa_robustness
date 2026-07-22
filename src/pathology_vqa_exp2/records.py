from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


OPTION_LETTERS = ("A", "B", "C", "D")
QUESTION_VARIANTS = ("canonical", "p1", "p2", "p3")
PROMPT_CONDITIONS = ("qa", "instruction")
NATIVE_VLM_MODEL_KEYS = ("internvl3", "qwen3_vl")


@dataclass(frozen=True)
class QuestionRecord:
    qid: str
    dataset: str
    source_id: str
    slide: str
    task: str
    question: str
    options: dict[str, str]
    answer: str
    paraphrases: dict[str, str]
    feature_path: Path | None = None
    image_paths: tuple[Path, ...] = ()

    @property
    def valid_options(self) -> tuple[str, ...]:
        return tuple(letter for letter in OPTION_LETTERS if self.options.get(letter))


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    model_key: str
    qid: str
    dataset: str
    source_id: str
    slide: str
    task: str
    question_variant: str
    prompt_condition: str
    question: str
    options: dict[str, str]
    gold_option: str
    system_prompt: str | None
    user_prompt: str
    feature_path: str | None
    image_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["image_paths"] = list(self.image_paths)
        for letter in OPTION_LETTERS:
            row[letter] = self.options.get(letter, "")
        del row["options"]
        return row


@dataclass(frozen=True)
class NativePatchCase:
    """One single-patch model call belonging to a slide-level experiment case."""

    case_id: str
    slide_case_id: str
    patch_index: int
    patch_name: str
    image_path: str
    base_case: ExperimentCase

    def model_case(self) -> ExperimentCase:
        """Return the single-image case passed to a native-VLM adapter."""
        return replace(
            self.base_case,
            case_id=self.case_id,
            image_paths=(self.image_path,),
        )

    def to_dict(self) -> dict[str, Any]:
        row = self.model_case().to_dict()
        row.update(
            {
                "slide_case_id": self.slide_case_id,
                "patch_index": self.patch_index,
                "patch_name": self.patch_name,
                "image_path": self.image_path,
            }
        )
        return row
