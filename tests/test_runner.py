import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from pathology_vqa_exp2.data import build_experiment_cases, load_questions
from pathology_vqa_exp2.models.base import ModelAdapter
from pathology_vqa_exp2.runner import run_cases


ROOT = Path(__file__).resolve().parents[1]


class FakeAdapter(ModelAdapter):
    model_key = "qwen3_text"
    model_id = "fake/qwen"

    def generate(self, case):
        return json.dumps({"answer": case.gold_option})


class RunnerTest(unittest.TestCase):
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
            checkpoint_lines = (output / "predictions.checkpoint.jsonl").read_text().splitlines()
            self.assertEqual(len(checkpoint_lines), 8)

            changed_cases = [
                replace(cases[0], user_prompt=cases[0].user_prompt + " changed"),
                *cases[1:],
            ]
            with self.assertRaisesRegex(ValueError, "user_prompt changed"):
                run_cases(FakeAdapter(), changed_cases, output, resume=True)

            second = run_cases(FakeAdapter(), cases, output, resume=True)
            self.assertEqual(second["already_completed"], 8)
            checkpoint_lines = (output / "predictions.checkpoint.jsonl").read_text().splitlines()
            self.assertEqual(len(checkpoint_lines), 8)


if __name__ == "__main__":
    unittest.main()
