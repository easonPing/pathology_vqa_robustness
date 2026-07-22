#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathology_vqa_exp2.cache import DEFAULT_HF_HOME
from pathology_vqa_exp2.data import build_experiment_cases, load_questions
from pathology_vqa_exp2.records import NATIVE_VLM_MODEL_KEYS
from pathology_vqa_exp2.runner import (
    DEFAULT_NATIVE_PATCH_COUNT,
    build_native_patch_cases,
    run_cases,
    run_native_patch_cases,
    write_cases,
    write_native_patch_cases,
)


DEFAULT_SUBSET = ROOT / "data/processed/exp2_question_subset.csv"
DEFAULT_PARAPHRASES = ROOT / "data/processed/vqa_paraphrases.json"
DEFAULT_MANIFEST = ROOT / "data/manifests/slidechat_feature_manifest.csv"
DEFAULT_IMAGE_MANIFEST = ROOT / "data/manifests/native_patch_manifest.csv"
MODELS_CONFIG_PATH = ROOT / "configs/models_config/models.json"
MODELS_CONFIG = json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
NATIVE_VLM_MODELS = set(NATIVE_VLM_MODEL_KEYS)


def configured_model_id(model_key: str) -> str:
    value = MODELS_CONFIG.get(model_key, {}).get("model_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing model_id for {model_key} in {MODELS_CONFIG_PATH}")
    return value.strip()


def configured_model_revision(model_key: str) -> str:
    value = MODELS_CONFIG.get(model_key, {}).get("revision")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing revision for {model_key} in {MODELS_CONFIG_PATH}")
    return value.strip()


def configured_repo_path(model_key: str, field: str) -> Path:
    value = MODELS_CONFIG.get(model_key, {}).get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Missing {field} for {model_key} in {MODELS_CONFIG_PATH}"
        )
    return (ROOT / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled pathology VQA perturbation inference."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("qwen3_text", "slidechat", "internvl3", "qwen3_vl"),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--qids", nargs="+", help="Explicit frozen question IDs")
    selection.add_argument("--all", action="store_true", help="Run all frozen questions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--paraphrases", type=Path, default=DEFAULT_PARAPHRASES)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-manifest", type=Path, default=DEFAULT_IMAGE_MANIFEST)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument(
        "--slidechat-repo",
        type=Path,
        default=configured_repo_path("slidechat", "source_path"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=configured_repo_path("slidechat", "checkpoint_path"),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and write all cases without loading a model",
    )
    return parser.parse_args()


def build_adapter(args: argparse.Namespace):
    if args.model == "qwen3_text":
        from pathology_vqa_exp2.models.qwen_text import QwenTextAdapter

        return QwenTextAdapter(
            model_id=args.model_id or configured_model_id("qwen3_text"),
            hf_home=args.hf_home,
            device=args.device,
            dtype=args.dtype or "bfloat16",
            max_new_tokens=args.max_new_tokens,
            revision=args.revision or configured_model_revision("qwen3_text"),
        )

    if args.model == "qwen3_vl":
        from pathology_vqa_exp2.models.qwen_vl import QwenVLAdapter

        return QwenVLAdapter(
            model_id=args.model_id or configured_model_id("qwen3_vl"),
            hf_home=args.hf_home,
            device=args.device,
            dtype=args.dtype or "bfloat16",
            max_new_tokens=args.max_new_tokens,
            revision=args.revision or configured_model_revision("qwen3_vl"),
        )

    if args.model == "internvl3":
        from pathology_vqa_exp2.models.internvl import InternVLAdapter

        return InternVLAdapter(
            model_id=args.model_id or configured_model_id("internvl3"),
            hf_home=args.hf_home,
            device=args.device,
            dtype=args.dtype or "bfloat16",
            max_new_tokens=args.max_new_tokens,
            image_size=args.image_size,
            revision=args.revision or configured_model_revision("internvl3"),
        )

    if not args.slidechat_repo.is_dir():
        raise SystemExit(f"SlideChat repository does not exist: {args.slidechat_repo}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"SlideChat checkpoint does not exist: {args.checkpoint}")
    from pathology_vqa_exp2.models.slidechat import SlideChatAdapter

    return SlideChatAdapter(
        slidechat_repo=args.slidechat_repo,
        checkpoint=args.checkpoint,
        model_id=args.model_id or configured_model_id("slidechat"),
        hf_home=args.hf_home,
        device=args.device,
        dtype=args.dtype or "float16",
        max_new_tokens=args.max_new_tokens,
        revision=args.revision or configured_model_revision("slidechat"),
    )


def main() -> None:
    args = parse_args()
    image_manifest = None
    if args.model in NATIVE_VLM_MODELS:
        if not args.image_manifest.is_file():
            raise SystemExit(
                f"Native-VLM inference requires an image manifest: "
                f"{args.image_manifest}"
            )
        image_manifest = args.image_manifest
    requires_images = args.model in NATIVE_VLM_MODELS and image_manifest is not None
    records = load_questions(
        args.subset,
        args.paraphrases,
        feature_manifest_path=(
            args.feature_manifest if args.model == "slidechat" else None
        ),
        require_features=args.model == "slidechat",
        image_manifest_path=image_manifest,
        require_images=requires_images,
        images_per_question=(
            DEFAULT_NATIVE_PATCH_COUNT if image_manifest is not None else None
        ),
        qids=None if args.all else args.qids,
    )
    cases = build_experiment_cases(records, args.model)
    print(
        f"selected {len(records)} questions and {len(cases)} cases "
        f"for {args.model}"
    )
    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / "cases.csv"
        write_cases(path, cases)
        print(f"wrote: {path}")
        if args.model in NATIVE_VLM_MODELS:
            patch_cases = build_native_patch_cases(
                cases,
                expected_patch_count=DEFAULT_NATIVE_PATCH_COUNT,
            )
            patch_path = args.output_dir / "patch_cases.csv"
            write_native_patch_cases(patch_path, patch_cases)
            print(f"wrote: {patch_path}")
        return

    adapter = build_adapter(args)
    if args.model in NATIVE_VLM_MODELS:
        run_native_patch_cases(
            adapter,
            cases,
            args.output_dir,
            resume=not args.no_resume,
            fail_fast=args.fail_fast,
        )
    else:
        run_cases(
            adapter,
            cases,
            args.output_dir,
            resume=not args.no_resume,
            fail_fast=args.fail_fast,
        )


if __name__ == "__main__":
    main()
