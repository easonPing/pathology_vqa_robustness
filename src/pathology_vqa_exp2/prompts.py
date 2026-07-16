from __future__ import annotations

from .records import OPTION_LETTERS


EXPERT_SYSTEM_PROMPT = "You are an expert pathologist."
QWEN_MCQ_INSTRUCTION = (
    "Return your choice as JSON with only the option letter in the answer "
    'field, for example: {"answer": "C"}.'
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
        return None, f"Question:\n{question}\n{options_text}\nAnswer:"

    system_prompt = EXPERT_SYSTEM_PROMPT
    if model_key in {"qwen3_text", "qwen3_vl"}:
        user_prompt = f"{question}\n{options_text}\n\n{QWEN_MCQ_INSTRUCTION}"
    elif model_key == "slidechat":
        # This is SlideChat's official test-time input format.
        user_prompt = f"{question}\n{options_text}"
    elif model_key == "internvl3":
        user_prompt = (
            f"{question}\n{options_text}\n\n{INTERNVL_MCQ_INSTRUCTION}"
        )
    else:
        raise ValueError(f"No instruction prompt is defined for model: {model_key}")

    return system_prompt, user_prompt
