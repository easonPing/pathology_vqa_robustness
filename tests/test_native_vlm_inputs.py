import csv
import tempfile
import unittest
from pathlib import Path

from pathology_vqa_exp2.data import build_experiment_cases, load_questions
from pathology_vqa_exp2.images import load_native_image_manifest
from pathology_vqa_exp2.models.internvl import InternVLAdapter
from pathology_vqa_exp2.models.qwen_vl import QwenVLAdapter
from pathology_vqa_exp2.prompts import (
    EXPERT_SYSTEM_PROMPT,
    INTERNVL_MCQ_INSTRUCTION,
)


ROOT = Path(__file__).resolve().parents[1]


class NativeVLMInputTest(unittest.TestCase):
    def make_manifest(self, directory: Path, *, second_status: str = "found") -> Path:
        image_paths = [directory / "patch_0.jpg", directory / "patch_1.jpg"]
        for image_path in image_paths:
            image_path.write_bytes(b"test image placeholder")
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
                            "" if index == 1 and second_status == "missing" else image_path
                        ),
                        "sha256": (
                            ""
                            if index == 1 and second_status == "missing"
                            else f"{index:064x}"
                        ),
                        "status": second_status if index == 1 else "found",
                    }
                )
        return manifest

    def load_record(self, manifest: Path):
        return load_questions(
            ROOT / "data/processed/exp2_question_subset.csv",
            ROOT / "data/processed/vqa_paraphrases.json",
            image_manifest_path=manifest,
            require_images=True,
            images_per_question=2,
            qids=["TCGA-1"],
        )[0]

    def test_manifest_images_are_frozen_across_all_eight_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self.load_record(self.make_manifest(Path(directory)))
            cases = build_experiment_cases([record], "qwen3_vl")
            self.assertEqual(len(cases), 8)
            self.assertEqual(len({case.image_paths for case in cases}), 1)
            self.assertEqual(len(cases[0].image_paths), 2)

    def test_qwen_messages_bind_system_only_to_instruction_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self.load_record(self.make_manifest(Path(directory)))
            cases = build_experiment_cases([record], "qwen3_vl")
            qa = next(case for case in cases if case.case_id.endswith("::qa"))
            instruction = next(
                case for case in cases if case.case_id.endswith("::instruction")
            )
            qa_messages = QwenVLAdapter.build_messages(qa)
            instruction_messages = QwenVLAdapter.build_messages(instruction)
            self.assertEqual([message["role"] for message in qa_messages], ["user"])
            self.assertEqual(instruction_messages[0]["role"], "system")
            self.assertEqual(
                instruction_messages[0]["content"], EXPERT_SYSTEM_PROMPT
            )
            image_items = instruction_messages[-1]["content"][:-1]
            self.assertEqual(len(image_items), 2)

    def test_internvl_instruction_and_multi_image_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self.load_record(self.make_manifest(Path(directory)))
            cases = build_experiment_cases([record], "internvl3")
            instruction = next(
                case for case in cases if case.case_id.endswith("::instruction")
            )
            question = InternVLAdapter.build_question(instruction)
            self.assertTrue(question.startswith("Image-1: <image>\nImage-2: <image>\n"))
            self.assertIn(INTERNVL_MCQ_INSTRUCTION, question)
            self.assertEqual(instruction.system_prompt, EXPERT_SYSTEM_PROMPT)

    def test_missing_selected_patch_fails_strict_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(Path(directory), second_status="missing")
            with self.assertRaisesRegex(ValueError, "missing patches"):
                self.load_record(manifest)

    def test_qids_on_same_slide_must_share_ordered_patches(self):
        with tempfile.TemporaryDirectory() as directory:
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
                    image_path.write_bytes(b"test image placeholder")
                    writer.writerow(
                        {
                            "qid": qid,
                            "dataset": "TCGA",
                            "slide": "same-slide",
                            "patch_index": 0,
                            "patch_name": image_path.name,
                            "image_path": image_path,
                            "sha256": f"{index:064x}",
                            "status": "found",
                        }
                    )
            with self.assertRaisesRegex(ValueError, "identical ordered patches"):
                load_native_image_manifest(manifest, repo_root=ROOT)


if __name__ == "__main__":
    unittest.main()
