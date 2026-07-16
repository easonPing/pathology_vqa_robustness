#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathology_vqa_exp2.cache import DEFAULT_HF_HOME
from pathology_vqa_exp2.data import build_experiment_cases, load_questions
from pathology_vqa_exp2.runner import run_cases, write_cases


DEFAULT_SUBSET = ROOT / "data/processed/exp2_question_subset.csv"
DEFAULT_PARAPHRASES = ROOT / "data/processed/vqa_paraphrases.json"
DEFAULT_MANIFEST = (
    ROOT / "data/features/survey_prompt_feature/exp2_feature_manifest.csv"
)
DEFAULT_IMAGE_MANIFEST = ROOT / "data/images/native_patch_manifest.csv"
NATIVE_VLM_MODELS = {"internvl3", "qwen3_vl"}


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
    parser.add_argument("--images-per-question", type=int, default=16)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--slidechat-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
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
            model_id=args.model_id or "Qwen/Qwen3-8B",
            hf_home=args.hf_home,
            device=args.device,
            dtype=args.dtype or "bfloat16",
            max_new_tokens=args.max_new_tokens,
            revision=args.revision,
        )

    if args.model == "qwen3_vl":
        from pathology_vqa_exp2.models.qwen_vl import QwenVLAdapter

        return QwenVLAdapter(
            model_id=args.model_id or "Qwen/Qwen3-VL-8B-Instruct",
            hf_home=args.hf_home,
            device=args.device,
            dtype=args.dtype or "bfloat16",
            max_new_tokens=args.max_new_tokens,
            revision=args.revision,
        )

    if args.model == "internvl3":
        from pathology_vqa_exp2.models.internvl import InternVLAdapter

        return InternVLAdapter(
            model_id=args.model_id or "OpenGVLab/InternVL3-8B",
            hf_home=args.hf_home,
            device=args.device,
            dtype=args.dtype or "bfloat16",
            max_new_tokens=args.max_new_tokens,
            image_size=args.image_size,
            revision=args.revision,
        )

    if args.slidechat_repo is None or args.checkpoint is None:
        raise SystemExit(
            "SlideChat requires both --slidechat-repo and --checkpoint"
        )
    from pathology_vqa_exp2.models.slidechat import SlideChatAdapter

    return SlideChatAdapter(
        slidechat_repo=args.slidechat_repo,
        checkpoint=args.checkpoint,
        model_id=args.model_id or "Qwen/Qwen2.5-7B-Instruct",
        hf_home=args.hf_home,
        device=args.device,
        dtype=args.dtype or "float16",
        max_new_tokens=args.max_new_tokens,
        revision=args.revision,
    )


def main() -> None:
    args = parse_args()
    image_manifest = (
        args.image_manifest
        if args.model in NATIVE_VLM_MODELS and args.image_manifest.is_file()
        else None
    )
    if args.model in NATIVE_VLM_MODELS and not args.dry_run:
        if image_manifest is None:
            raise SystemExit(
                f"Native-VLM inference requires an image manifest: "
                f"{args.image_manifest}"
            )
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
            args.images_per_question if image_manifest is not None else None
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
        return

    adapter = build_adapter(args)
    run_cases(
        adapter,
        cases,
        args.output_dir,
        resume=not args.no_resume,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
