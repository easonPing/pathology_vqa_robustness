import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import pandas as pd

from pathology_vqa_exp2.data import build_experiment_cases, load_questions
from pathology_vqa_exp2.metrics import validate_predictions
from pathology_vqa_exp2.models.base import ModelAdapter
from pathology_vqa_exp2.runner import (
    aggregate_native_patch_predictions,
    build_native_patch_cases,
    run_cases,
    run_native_patch_cases,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeAdapter(ModelAdapter):
    model_key = "qwen3_text"
    model_id = "fake/qwen"

    def generate(self, case):
        return json.dumps({"answer": case.gold_option})


class FakeNativeAdapter(ModelAdapter):
    model_key = "qwen3_vl"
    model_id = "fake/qwen-vl"

    def __init__(self):
        self.calls = 0

    def generate(self, case):
        self.calls += 1
        if len(case.image_paths) != 1:
            raise AssertionError("native adapter received a multi-image case")
        patch_index = int(case.case_id.rsplit("::", 1)[1])
        return json.dumps({"answer": "A" if patch_index < 16 else "B"})


class RunnerTest(unittest.TestCase):
    def load_one_case_set(self, model_key):
        record = load_questions(
            ROOT / "data/processed/exp2_question_subset.csv",
            ROOT / "data/processed/vqa_paraphrases.json",
            qids=["TCGA-1"],
        )[0]
        return build_experiment_cases([record], model_key)

    def test_checkpoint_final_csv_and_resume(self):
        record = load_questions(
            ROOT / "data/processed/exp2_question_subset.csv",
            ROOT / "data/processed/vqa_paraphrases.json",
            qids=["TCGA-1"],
        )[0]
        cases = build_experiment_cases([record], "qwen3_text")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = run_cases(FakeAdapter(), cases, output)
            self.assertEqual(first["completed_count"], 8)
            predictions = pd.read_csv(output / "predictions.csv")
            self.assertEqual(len(predictions), 8)
            self.assertTrue(predictions["correct"].all())
            checkpoint_lines = (
                output / "predictions.checkpoint.jsonl"
            ).read_text().splitlines()
            self.assertEqual(len(checkpoint_lines), 8)

            changed_cases = [
                replace(cases[0], user_prompt=cases[0].user_prompt + " changed"),
                *cases[1:],
            ]
            with self.assertRaisesRegex(ValueError, "user_prompt changed"):
                run_cases(FakeAdapter(), changed_cases, output, resume=True)

            second = run_cases(FakeAdapter(), cases, output, resume=True)
            self.assertEqual(second["already_completed"], 8)
            checkpoint_lines = (
                output / "predictions.checkpoint.jsonl"
            ).read_text().splitlines()
            self.assertEqual(len(checkpoint_lines), 8)

    def test_native_runner_writes_240_patch_rows_and_8_slide_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            image_paths = tuple(
                str(output / f"patch_{index}.jpg") for index in range(30)
            )
            cases = [
                replace(case, image_paths=image_paths)
                for case in self.load_one_case_set("qwen3_vl")
            ]
            adapter = FakeNativeAdapter()
            with redirect_stdout(io.StringIO()):
                first = run_native_patch_cases(adapter, cases, output)

            self.assertEqual(adapter.calls, 240)
            self.assertEqual(first["patch_call_count"], 240)
            self.assertEqual(first["patches_per_case"], 30)
            self.assertEqual(
                first["aggregation_policy"], "five_category_unique_mode_v1"
            )
            patch_predictions = pd.read_csv(output / "patch_predictions.csv")
            predictions = pd.read_csv(output / "predictions.csv")
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "cases.csv",
                    "patch_cases.csv",
                    "patch_predictions.checkpoint.jsonl",
                    "patch_predictions.csv",
                    "predictions.csv",
                    "run_metadata.json",
                },
            )
            self.assertEqual(len(patch_predictions), 240)
            self.assertEqual(len(predictions), 8)
            self.assertTrue(
                (patch_predictions["image_paths"].str.count("patch_") == 1).all()
            )
            self.assertTrue((predictions["predicted_option"] == "A").all())
            self.assertTrue((predictions["vote_A"] == 16).all())
            self.assertTrue((predictions["vote_B"] == 14).all())
            self.assertTrue((predictions["valid_vote_count"] == 30).all())
            self.assertTrue((predictions["invalid_vote_count"] == 0).all())
            self.assertTrue((predictions["vote_INVALID"] == 0).all())
            self.assertTrue(
                (predictions["vote_status"] == "unique_valid_winner").all()
            )
            self.assertFalse(predictions["vote_tied"].all())
            validate_predictions(predictions)

            with redirect_stdout(io.StringIO()):
                second = run_native_patch_cases(adapter, cases, output, resume=True)
            self.assertEqual(adapter.calls, 240)
            self.assertEqual(second["already_completed_patch_calls"], 240)
            checkpoint_lines = (
                output / "patch_predictions.checkpoint.jsonl"
            ).read_text().splitlines()
            self.assertEqual(len(checkpoint_lines), 240)

            metadata_path = output / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["aggregation"]["patches_per_case"] = 29
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed aggregation"):
                run_native_patch_cases(adapter, cases, output, resume=True)

    def test_native_vote_counts_invalid_and_marks_non_valid_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            image_paths = tuple(
                str(Path(directory) / f"patch_{index}.jpg") for index in range(30)
            )
            case = replace(
                self.load_one_case_set("qwen3_vl")[0],
                image_paths=image_paths,
            )
            patch_cases = build_native_patch_cases([case])

            def aggregate(votes):
                rows = {
                    patch_case.case_id: {
                        "status": "ok",
                        "predicted_option": vote,
                        "latency_seconds": 0.1,
                    }
                    for patch_case, vote in zip(patch_cases, votes)
                }
                return aggregate_native_patch_predictions(
                    [case], rows, model_id="fake/qwen-vl"
                )[0]

            unique = aggregate(["A"] * 13 + ["B"] * 11 + ["INVALID"] * 6)
            self.assertEqual(unique["predicted_option"], "A")
            self.assertEqual(unique["valid_vote_count"], 24)
            self.assertEqual(unique["invalid_vote_count"], 6)
            self.assertEqual(unique["vote_INVALID"], 6)
            self.assertEqual(unique["vote_status"], "unique_valid_winner")
            self.assertFalse(unique["vote_tied"])

            invalid_winner = aggregate(
                ["A"] * 5 + ["B"] * 4 + ["INVALID"] * 21
            )
            self.assertEqual(invalid_winner["predicted_option"], "INVALID")
            self.assertEqual(
                invalid_winner["parse_method"],
                "majority_vote_invalid_winner",
            )
            self.assertEqual(invalid_winner["vote_status"], "invalid_winner")
            self.assertFalse(invalid_winner["vote_tied"])

            tied = aggregate(["A"] * 12 + ["B"] * 12 + ["INVALID"] * 6)
            self.assertEqual(tied["predicted_option"], "INVALID")
            self.assertEqual(tied["parse_method"], "majority_vote_tie_invalid")
            self.assertEqual(tied["vote_status"], "tie")
            self.assertTrue(tied["vote_tied"])
            self.assertFalse(tied["correct"])

            no_valid = aggregate(["INVALID"] * 30)
            self.assertEqual(no_valid["predicted_option"], "INVALID")
            self.assertEqual(
                no_valid["parse_method"], "majority_vote_no_valid_votes"
            )
            self.assertEqual(no_valid["valid_vote_count"], 0)
            self.assertEqual(no_valid["invalid_vote_count"], 30)
            self.assertEqual(no_valid["vote_status"], "no_valid")

    def test_native_expansion_requires_exactly_30_unique_patches(self):
        case = replace(
            self.load_one_case_set("qwen3_vl")[0],
            image_paths=tuple(f"patch_{index}.jpg" for index in range(29)),
        )
        with self.assertRaisesRegex(ValueError, "expected exactly 30 patches"):
            build_native_patch_cases([case])
        duplicate = replace(case, image_paths=("same.jpg",) * 30)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            build_native_patch_cases([duplicate])


if __name__ == "__main__":
    unittest.main()
