from __future__ import annotations

from .records import OPTION_LETTERS


EXPERT_SYSTEM_PROMPT = "You are an expert pathologist."
QA_JSON_ANSWER_INSTRUCTION = (
    "Return your choice as JSON with only the option letter in the answer "
    'field, for example: {"answer": "C"}.'
)
QWEN3_TEXT_MCQ_INSTRUCTION = (
    "Please show your choice in the answer field with only the choice letter, "
    'e.g., "answer": "C".'
)
QWEN3_VL_MCQ_INSTRUCTION = (
    "Please select the correct answer from the options above."
)
INTERNVL_MCQ_INSTRUCTION = (
    "Answer with the option's letter from the given choices directly."
)


def format_options(options: dict[str, str]) -> str:
    return "\n".join(
        f"{letter}. {options[letter]}"
        for letter in OPTION_LETTERS
        if options.get(letter)
    )


def render_prompt(
    model_key: str,
    question: str,
    options: dict[str, str],
    prompt_condition: str,
) -> tuple[str | None, str]:
    if prompt_condition not in {"qa", "instruction"}:
        raise ValueError(f"Unknown prompt condition: {prompt_condition}")

    options_text = format_options(options)

    if prompt_condition == "qa":
        user_prompt = (
            f"Question:\n{question}\n{options_text}\n"
            f"{QA_JSON_ANSWER_INSTRUCTION}\nAnswer:"
        )
        return None, user_prompt

    system_prompt = EXPERT_SYSTEM_PROMPT
    if model_key == "qwen3_text":
        # Qwen3's official model card recommends this MCQ output instruction.
        user_prompt = (
            f"{question}\n{options_text}\n{QWEN3_TEXT_MCQ_INSTRUCTION}\n"
            f"{QA_JSON_ANSWER_INSTRUCTION}\nAnswer:"
        )
    elif model_key == "qwen3_vl":
        # Qwen3-VL's official VLMEvalKit configuration uses the dataset's
        # default ImageMCQDataset prompt rather than a model-specific override.
        user_prompt = (
            f"Question: {question}\nOptions:\n{options_text}\n"
            f"{QWEN3_VL_MCQ_INSTRUCTION}\n"
            f"{QA_JSON_ANSWER_INSTRUCTION}\nAnswer:"
        )
    elif model_key == "slidechat":
        # This is SlideChat's official test-time input format.
        user_prompt = f"{question}\n{options_text}"
    elif model_key == "internvl3":
        user_prompt = f"{question}\n{options_text}\n{INTERNVL_MCQ_INSTRUCTION}"
    else:
        raise ValueError(f"No instruction prompt is defined for model: {model_key}")

    return system_prompt, user_prompt
