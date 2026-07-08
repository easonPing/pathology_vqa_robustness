
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


OPTION_COLUMNS = ("A", "B", "C", "D")


def read_subset(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"qid", "dataset", "ID", "Slide", "Task", "Question", "A", "B", "C", "D", "Answer"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return rows


def non_empty_options(row: Mapping[str, str]) -> List[str]:
    return [letter for letter in OPTION_COLUMNS if row.get(letter, "").strip()]


def format_options(row: Mapping[str, str]) -> str:
    return "\n".join(f"{letter}. {row[letter]}" for letter in non_empty_options(row))


def make_item(row: Mapping[str, str], paraphrases: Sequence[str]) -> dict:
    return {
        "qid": row["qid"],
        "dataset": row["dataset"],
        "ID": row["ID"],
        "Slide": row["Slide"],
        "Task": row["Task"],
        "Question": row["Question"],
        "options": {letter: row.get(letter, "") for letter in OPTION_COLUMNS},
        "non_empty_options": non_empty_options(row),
        "Answer": row["Answer"],
        "paraphrases": [
            {
                "paraphrase_id": f"{row['qid']}-p{i + 1}",
                "Question": question,
            }
            for i, question in enumerate(paraphrases)
        ],
    }


def build_paraphrase_prompt(
    row: Mapping[str, str],
    accepted: Sequence[str],
    needed_count: int,
) -> str:
    example_items = ",\n    ".join(
        f'"Paraphrased question {index}."' for index in range(1, needed_count + 1)
    )
    previous = ""
    if accepted:
        previous = "\nThese are already accepted paraphrases. Do not repeat them:\n" + "\n".join(
            f"{question}" for question in accepted
        )
    return f"""Rewrite and return exactly {needed_count} paraphrases of the original pathology VQA question below while preserving its semantic meaning.

Rules:
- Return strict JSON with exactly one key: "Paraphrases". The value of "Paraphrases" must be a JSON array of exactly {needed_count} strings. Each string must be one paraphrased question.
- Do not provide the answer options.
- Do not provide explanations, labels, or any text outside the JSON object.
- Preserve the clinical entities, queried attributes, and the semantic meaning of the original question.
- Make the paraphrases different from the original question and from already accepted paraphrases.

Output format:
{{
  "Paraphrases": [
    {example_items}
  ]
}}

Original question:
{row['Question']}

Options for context only. Do not rewrite them:
{format_options(row)}

Gold answer for context only. Do not mention it:
{row['Answer']}
{previous}

JSON output:"""


def build_check_prompt(row: Mapping[str, str], paraphrase: str) -> str:
    return f"""The candidate question is a candidate paraphrase of the original pathology VQA question.

Check whether the clinical entities, the queried attributes, and the semantic meaning of the original question are unchanged in the candidate paraphrases.

Only evaluate the question text. Do not evaluate or rewrite the options or answer.

Return strict JSON with exactly these keys:
- clinical_entities_unchanged: boolean
- queried_attributes_unchanged: boolean
- semantic_meaning_unchanged: boolean
- reason: short string

Original question:
{row['Question']}

Candidate question:
{paraphrase}

Options for context only:
{format_options(row)}

Gold answer for context only:
{row['Answer']}
"""


def normalize_for_duplicate_check(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    return text


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def parse_paraphrases_json(text: str) -> List[str]:
    payload = extract_json_object(text)
    paraphrases = payload.get("Paraphrases")
    if not isinstance(paraphrases, list):
        raise ValueError('Generator JSON must contain a "Paraphrases" array')

    parsed = []
    for paraphrase in paraphrases:
        if not isinstance(paraphrase, str):
            raise ValueError('Every item in "Paraphrases" must be a string')
        cleaned = paraphrase.strip().strip("\"'`").strip()
        if cleaned:
            parsed.append(cleaned)
    return parsed


def passes_semantic_check(verdict: Mapping[str, Any]) -> bool:
    return all(
        verdict.get(key) is True
        for key in (
            "clinical_entities_unchanged",
            "queried_attributes_unchanged",
            "semantic_meaning_unchanged",
        )
    )


class QwenChatModel:
    def __init__(
        self,
        model_id: str,
        device_map: str,
        dtype: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        trust_remote_code: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen generation requires torch and transformers. Install them before running paraphrase generation."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch_dtype,
                **model_kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer([text], return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        generation_temperature = self.temperature if temperature is None else temperature
        generation_top_p = self.top_p if top_p is None else top_p
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": generation_temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if generation_temperature > 0:
            generation_kwargs["temperature"] = generation_temperature
            generation_kwargs["top_p"] = generation_top_p
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def generate_checked_paraphrases(
    row: Mapping[str, str],
    generator: QwenChatModel,
    checker: QwenChatModel,
    k: int,
    max_attempts: int,
) -> List[str]:
    accepted: List[str] = []
    seen = {normalize_for_duplicate_check(row["Question"])}
    last_error = "no attempts made"

    for _ in range(max_attempts):
        needed_count = k - len(accepted)
        raw = generator.generate(
            build_paraphrase_prompt(row, accepted, needed_count=needed_count),
            system_prompt="You are a helpful pathology VQA rewrite assistant.",
        )
        try:
            candidates = parse_paraphrases_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"generator did not return valid Paraphrases JSON: {exc}"
            continue

        if not candidates:
            last_error = "generator returned an empty Paraphrases array"
            continue

        for candidate in candidates:
            normalized = normalize_for_duplicate_check(candidate)
            if not candidate or normalized in seen:
                last_error = "empty, unchanged, or duplicate paraphrase"
                continue
            if len(candidate.split()) < 3:
                last_error = "candidate too short"
                continue

            check_raw = checker.generate(
                build_check_prompt(row, candidate),
                system_prompt="You are a strict pathology VQA semantic equivalence checker.",
                temperature=0.0,
                top_p=1.0,
            )
            try:
                verdict = extract_json_object(check_raw)
            except json.JSONDecodeError:
                last_error = f"checker did not return parseable JSON: {check_raw[:200]}"
                continue
            if passes_semantic_check(verdict):
                accepted.append(candidate)
                seen.add(normalized)
                if len(accepted) == k:
                    return accepted
            else:
                last_error = str(verdict.get("reason", "semantic check failed"))

    raise RuntimeError(
        f"{row['qid']} failed to produce {k} valid paraphrases "
        f"after {max_attempts} attempts: {last_error}"
    )


def generate_items(
    rows: Sequence[Mapping[str, str]],
    generator: QwenChatModel,
    checker: QwenChatModel,
    k: int,
    max_attempts: int,
    checkpoint: Path | None,
    metadata: Mapping[str, Any],
) -> List[dict]:
    items = []
    for index, row in enumerate(rows, start=1):
        accepted = generate_checked_paraphrases(
            row,
            generator=generator,
            checker=checker,
            k=k,
            max_attempts=max_attempts,
        )
        items.append(make_item(row, accepted))
        print(f"[{index}/{len(rows)}] {row['qid']}: accepted {len(accepted)} paraphrases")
        if checkpoint is not None:
            write_json(checkpoint, {"metadata": dict(metadata), "items": items})
    return items


def largest_remainder_quotas(counts: Mapping[str, int], target: int, min_one: bool = False) -> Dict[str, int]:
    if target <= 0:
        return {key: 0 for key in counts}
    if min_one and target >= len(counts):
        quotas = {key: 1 for key in counts}
        target -= len(counts)
    else:
        quotas = {key: 0 for key in counts}

    if target == 0:
        return quotas
    total = sum(counts.values())
    raw = {key: counts[key] / total * target for key in counts}
    base = {key: int(value) for key, value in raw.items()}
    remainder = target - sum(base.values())
    order = sorted(
        counts,
        key=lambda key: (raw[key] - base[key], counts[key], str(key)),
        reverse=True,
    )
    for key in counts:
        quotas[key] += base[key]
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def stratified_audit_sample(items: Sequence[dict], sample_n: int, seed: int) -> List[dict]:
    if sample_n >= len(items):
        return list(items)
    rng = random.Random(seed)
    groups = defaultdict(list)
    for item in items:
        groups[(item["dataset"], item["Task"])].append(item)
    quotas = largest_remainder_quotas(
        {key: len(value) for key, value in groups.items()},
        sample_n,
        min_one=True,
    )
    sampled = []
    for key in sorted(groups):
        group_items = list(groups[key])
        rng.shuffle(group_items)
        sampled.extend(group_items[: quotas[key]])
    sampled.sort(key=lambda item: (item["dataset"], item["Task"], item["qid"]))
    return sampled


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_audit_csv(path: Path, items: Sequence[dict]) -> None:
    fieldnames = [
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
        "paraphrase_1",
        "paraphrase_2",
        "paraphrase_3",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            row = {
                "qid": item["qid"],
                "dataset": item["dataset"],
                "ID": item["ID"],
                "Slide": item["Slide"],
                "Task": item["Task"],
                "Question": item["Question"],
                "A": item["options"].get("A", ""),
                "B": item["options"].get("B", ""),
                "C": item["options"].get("C", ""),
                "D": item["options"].get("D", ""),
                "Answer": item["Answer"],
            }
            for index in range(3):
                key = f"paraphrase_{index + 1}"
                row[key] = item["paraphrases"][index]["Question"] if index < len(item["paraphrases"]) else ""
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/exp2_question_subset.csv"))
    parser.add_argument("--output", type=Path, default=Path("vqa_paraphrases.json"))
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("exp2_question_paraphrased_sample_50.csv"),
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("vqa_paraphrases.checkpoint.json"))
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--checker-model-id", default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--audit-sample-n", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows for local smoke tests.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto", help="auto, float16, bfloat16, or float32")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k != 3:
        raise ValueError("This experiment requires --k 3")
    set_seed(args.seed)

    rows = read_subset(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    checker_model_id = args.checker_model_id or args.model_id
    metadata = {
        "source_subset": str(args.input),
        "generator_model": args.model_id,
        "checker_model": checker_model_id,
        "k": args.k,
        "seed": args.seed,
    }

    generator = QwenChatModel(
        model_id=args.model_id,
        device_map=args.device_map,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        trust_remote_code=args.trust_remote_code,
    )
    checker = generator
    if checker_model_id != args.model_id:
        checker = QwenChatModel(
            model_id=checker_model_id,
            device_map=args.device_map,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            trust_remote_code=args.trust_remote_code,
        )

    items = generate_items(
        rows,
        generator=generator,
        checker=checker,
        k=args.k,
        max_attempts=args.max_attempts,
        checkpoint=args.checkpoint,
        metadata=metadata,
    )
    payload = {"metadata": metadata, "items": items}
    write_json(args.output, payload)
    audit_items = stratified_audit_sample(items, args.audit_sample_n, seed=args.seed)
    write_audit_csv(args.audit_output, audit_items)
    print(f"wrote: {args.output}")
    print(f"wrote: {args.audit_output}")


if __name__ == "__main__":
    main()
