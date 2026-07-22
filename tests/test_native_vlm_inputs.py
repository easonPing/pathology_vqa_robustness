import csv
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from pathology_vqa_exp2.data import build_experiment_cases, load_questions
from pathology_vqa_exp2.images import load_native_image_manifest
from pathology_vqa_exp2.models.internvl import InternVLAdapter
from pathology_vqa_exp2.models.qwen_vl import QwenVLAdapter
from pathology_vqa_exp2.prompts import (
    EXPERT_SYSTEM_PROMPT,
    INTERNVL_MCQ_INSTRUCTION,
)
from pathology_vqa_exp2.runner import build_native_patch_cases


ROOT = Path(__file__).resolve().parents[1]


class NativeVLMInputTest(unittest.TestCase):
    def make_manifest(
        self,
        directory: Path,
        *,
        patch_count: int = 2,
        second_status: str = "found",
    ) -> Path:
        image_paths = [
            directory / f"patch_{index}.jpg" for index in range(patch_count)
        ]
        for image_path in image_paths:
            Image.new("RGB", (224, 224), color=(240, 200, 180)).save(image_path)
        manifest = directory / "native_patch_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "qid",
                    "dataset",
                    "slide",
                    "patch_index",
                    "patch_name",
                    "image_path",
                    "sha256",
                    "status",
                ],
            )
            writer.writeheader()
            for index, image_path in enumerate(image_paths):
                writer.writerow(
                    {
                        "qid": "TCGA-1",
                        "dataset": "TCGA",
                        "slide": "TCGA-OR-A5J8-01Z-00-DX1",
                        "patch_index": index,
                        "patch_name": image_path.name,
                        "image_path": (
                            ""
                            if index == 1 and second_status == "missing"
                            else image_path.relative_to(ROOT).as_posix()
                        ),
                        "sha256": (
                            ""
                            if index == 1 and second_status == "missing"
                            else hashlib.sha256(image_path.read_bytes()).hexdigest()
                        ),
                        "status": second_status if index == 1 else "found",
                    }
                )
        return manifest

    def test_native_dry_run_writes_slide_and_30_patch_case_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory_path = Path(directory)
            manifest = self.make_manifest(directory_path, patch_count=30)
            output = directory_path / "dry_run"
            command = [
                sys.executable,
                str(ROOT / "scripts/run_experiment.py"),
                "--model",
                "qwen3_vl",
                "--qids",
                "TCGA-1",
                "--output-dir",
                str(output),
                "--image-manifest",
                str(manifest),
                "--dry-run",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"cases.csv", "patch_cases.csv"},
            )
            self.assertEqual(len(pd.read_csv(output / "cases.csv")), 8)
            patch_cases = pd.read_csv(output / "patch_cases.csv")
            self.assertEqual(len(patch_cases), 240)
            self.assertEqual(patch_cases["slide_case_id"].nunique(), 8)
            self.assertTrue((patch_cases.groupby("slide_case_id").size() == 30).all())

    def load_record(self, manifest: Path):
        return load_questions(
            ROOT / "data/processed/exp2_question_subset.csv",
            ROOT / "data/processed/vqa_paraphrases.json",
            image_manifest_path=manifest,
            require_images=True,
            images_per_question=2,
            qids=["TCGA-1"],
        )[0]

    def test_manifest_images_expand_to_one_call_per_patch_for_all_eight_cases(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            record = self.load_record(self.make_manifest(Path(directory)))
            cases = build_experiment_cases([record], "qwen3_vl")
            self.assertEqual(len(cases), 8)
            self.assertEqual(len({case.image_paths for case in cases}), 1)
            self.assertEqual(len(cases[0].image_paths), 2)
            patch_cases = build_native_patch_cases(cases, expected_patch_count=2)
            self.assertEqual(len(patch_cases), 16)
            self.assertTrue(
                all(
                    len(patch_case.model_case().image_paths) == 1
                    for patch_case in patch_cases
                )
            )

    def test_qwen_messages_bind_system_only_to_instruction_condition(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            record = self.load_record(self.make_manifest(Path(directory)))
            cases = build_experiment_cases([record], "qwen3_vl")
            qa = next(case for case in cases if case.case_id.endswith("::qa"))
            instruction = next(
                case for case in cases if case.case_id.endswith("::instruction")
            )
            qa_patch = build_native_patch_cases(
                [qa], expected_patch_count=2
            )[0].model_case()
            instruction_patch = build_native_patch_cases(
                [instruction], expected_patch_count=2
            )[0].model_case()
            qa_messages = QwenVLAdapter.build_messages(qa_patch)
            instruction_messages = QwenVLAdapter.build_messages(instruction_patch)
            self.assertEqual([message["role"] for message in qa_messages], ["user"])
            self.assertEqual(instruction_messages[0]["role"], "system")
            self.assertEqual(
                instruction_messages[0]["content"],
                [{"type": "text", "text": EXPERT_SYSTEM_PROMPT}],
            )
            image_items = instruction_messages[-1]["content"][:-1]
            self.assertEqual(len(image_items), 1)
            with self.assertRaisesRegex(ValueError, "exactly one image"):
                QwenVLAdapter.build_messages(instruction)

    def test_internvl_instruction_and_single_image_prefix(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            record = self.load_record(self.make_manifest(Path(directory)))
            cases = build_experiment_cases([record], "internvl3")
            instruction = next(
                case for case in cases if case.case_id.endswith("::instruction")
            )
            patch_case = build_native_patch_cases(
                [instruction], expected_patch_count=2
            )[0].model_case()
            question = InternVLAdapter.build_question(patch_case)
            self.assertTrue(question.startswith("<image>\n"))
            self.assertIn(INTERNVL_MCQ_INSTRUCTION, question)
            self.assertEqual(instruction.system_prompt, EXPERT_SYSTEM_PROMPT)
            with self.assertRaisesRegex(ValueError, "exactly one image"):
                InternVLAdapter.build_question(instruction)

    def test_missing_selected_patch_fails_strict_loading(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            manifest = self.make_manifest(Path(directory), second_status="missing")
            with self.assertRaisesRegex(ValueError, "missing patches"):
                self.load_record(manifest)

    def test_manifest_sha_mismatch_fails_before_inference(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            manifest = self.make_manifest(Path(directory))
            frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
            image_path = ROOT / frame.iloc[0]["image_path"]
            Image.new("RGB", (224, 224), color=(1, 2, 3)).save(image_path)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.load_record(manifest)

    def test_qids_on_same_slide_must_share_ordered_patches(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory_path = Path(directory)
            manifest = directory_path / "native_patch_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "qid",
                        "dataset",
                        "slide",
                        "patch_index",
                        "patch_name",
                        "image_path",
                        "sha256",
                        "status",
                    ],
                )
                writer.writeheader()
                for index, qid in enumerate(("q1", "q2")):
                    image_path = directory_path / f"patch_{index}.jpg"
                    Image.new("RGB", (224, 224), color=(index, 20, 30)).save(
                        image_path
                    )
                    writer.writerow(
                        {
                            "qid": qid,
                            "dataset": "TCGA",
                            "slide": "same-slide",
                            "patch_index": 0,
                            "patch_name": image_path.name,
                            "image_path": image_path.relative_to(ROOT).as_posix(),
                            "sha256": hashlib.sha256(
                                image_path.read_bytes()
                            ).hexdigest(),
                            "status": "found",
                        }
                    )
            with self.assertRaisesRegex(ValueError, "identical ordered patches"):
                load_native_image_manifest(manifest, repo_root=ROOT)


if __name__ == "__main__":
    unittest.main()
