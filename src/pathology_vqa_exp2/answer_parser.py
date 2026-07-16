from __future__ import annotations

import json
import re
from dataclasses import dataclass


INVALID_OPTION = "INVALID"
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.I | re.S)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.I | re.S)


@dataclass(frozen=True)
class ParsedAnswer:
    option: str
    method: str


def _json_candidates(text: str) -> list[str]:
    candidates = _FENCED_JSON_RE.findall(text)
    candidates.extend(_JSON_OBJECT_RE.findall(text))
    return candidates


def _valid_letter(value: object, valid_options: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized if normalized in valid_options else None


def parse_answer(
    raw_output: str,
    options: dict[str, str],
) -> ParsedAnswer:
    valid_options = {letter for letter, text in options.items() if text.strip()}
    text = _THINK_RE.sub(" ", raw_output).strip()
    if not text:
        return ParsedAnswer(INVALID_OPTION, "empty")

    json_answers: set[str] = set()
    for candidate in _json_candidates(text):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        for key, answer in value.items():
            if str(key).lower() == "answer":
                letter = _valid_letter(answer, valid_options)
                if letter:
                    json_answers.add(letter)
    if len(json_answers) == 1:
        return ParsedAnswer(next(iter(json_answers)), "json")
    if len(json_answers) > 1:
        return ParsedAnswer(INVALID_OPTION, "ambiguous_json")

    explicit_patterns = (
        r"(?i)\b(?:final\s+)?answer\s*(?:is|:|=)?\s*\(?([A-D])\)?\b",
        r"(?i)\b(?:option|choice)\s*[:=]?\s*\(?([A-D])\)?\b",
    )
    explicit = {
        match.upper()
        for pattern in explicit_patterns
        for match in re.findall(pattern, text)
        if match.upper() in valid_options
    }
    if len(explicit) == 1:
        return ParsedAnswer(next(iter(explicit)), "explicit")
    if len(explicit) > 1:
        return ParsedAnswer(INVALID_OPTION, "ambiguous_explicit")

    leading = re.match(r"^\s*\(?([A-D])\)?(?:[.\]:-]|\s*$)", text, re.I)
    if leading and leading.group(1).upper() in valid_options:
        return ParsedAnswer(leading.group(1).upper(), "leading_letter")

    compact = text.strip().strip("`*#.:()[]{} \n\t").upper()
    if compact in valid_options:
        return ParsedAnswer(compact, "letter")

    normalized_output = re.sub(r"\s+", " ", text).strip().casefold()
    option_matches = {
        letter
        for letter, option_text in options.items()
        if letter in valid_options
        and normalized_output
        == re.sub(r"\s+", " ", option_text).strip().casefold()
    }
    if len(option_matches) == 1:
        return ParsedAnswer(next(iter(option_matches)), "option_text")

    return ParsedAnswer(INVALID_OPTION, "unparsed")
