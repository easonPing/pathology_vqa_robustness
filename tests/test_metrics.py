import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pathology_vqa_exp2.metrics import (
    compute_metrics,
    validate_predictions,
    write_metric_outputs,
)
from pathology_vqa_exp2.records import PROMPT_CONDITIONS, QUESTION_VARIANTS


def make_predictions():
    rows = []
    for qid in ("TCGA-test", "BCNB-test"):
        for variant in QUESTION_VARIANTS:
            for prompt in PROMPT_CONDITIONS:
                prediction = "C"
                if qid == "BCNB-test" and variant != "canonical":
                    prediction = "A"
                rows.append(
                    {
                        "case_id": f"{qid}::{variant}::{prompt}",
                        "model_key": "fake",
                        "model_id": "fake/model",
                        "qid": qid,
                        "dataset": qid.split("-")[0],
                        "question_variant": variant,
                        "prompt_condition": prompt,
                        "gold_option": "C",
                        "predicted_option": prediction,
                        "status": "ok",
                    }
                )
    return pd.DataFrame(rows)


class MetricsTest(unittest.TestCase):
    def test_three_perturbation_groups(self):
        metrics, per_question, positions = compute_metrics(
            make_predictions(), bootstrap_replicates=100, seed=42
        )
        self.assertEqual(
            set(metrics["perturbation"]),
            {"paraphrase", "prompt", "paraphrase+prompt"},
        )
        paraphrase = metrics[metrics["perturbation"] == "paraphrase"].iloc[0]
        self.assertEqual(paraphrase["canonical_accuracy"], 1.0)
        self.assertEqual(paraphrase["perturbed_accuracy"], 0.5)
        self.assertEqual(paraphrase["accuracy_delta"], 0.5)
        self.assertEqual(paraphrase["consistency"], 0.5)
        self.assertEqual(len(per_question), 6)
        self.assertEqual(len(positions), 15)
        self.assertEqual(paraphrase["perturbed_invalid_rate"], 0.0)
        self.assertTrue(math.isnan(paraphrase["perturbed_tie_rate"]))

    def test_native_invalid_and_tie_rates_are_reported(self):
        frame = make_predictions()
        frame["vote_status"] = "unique_valid_winner"
        target = (
            (frame["qid"] == "TCGA-test")
            & (frame["question_variant"] == "p1")
            & (frame["prompt_condition"] == "qa")
        )
        frame.loc[target, "predicted_option"] = "INVALID"
        frame.loc[target, "vote_status"] = "tie"
        metrics, _, _ = compute_metrics(
            frame, bootstrap_replicates=100, seed=42
        )
        paraphrase = metrics[metrics["perturbation"] == "paraphrase"].iloc[0]
        self.assertAlmostEqual(paraphrase["perturbed_invalid_rate"], 1 / 6)
        self.assertAlmostEqual(paraphrase["perturbed_tie_rate"], 1 / 6)

    def test_missing_combination_fails(self):
        frame = make_predictions().iloc[:-1]
        with self.assertRaisesRegex(ValueError, "expected 8"):
            validate_predictions(frame)

    def test_failed_inference_fails(self):
        frame = make_predictions()
        frame.loc[0, "status"] = "error"
        with self.assertRaisesRegex(ValueError, "inference cases failed"):
            validate_predictions(frame)

    def test_writes_all_metric_deliverables(self):
        with tempfile.TemporaryDirectory() as directory:
            write_metric_outputs(
                make_predictions(), directory, bootstrap_replicates=100, seed=42
            )
            output = Path(directory)
            expected = {
                "per_model_perturbation_metrics.csv",
                "per_question_metrics.csv",
                "position_bias.csv",
                "exp2_table4_row.md",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            table = (output / "exp2_table4_row.md").read_text()
            self.assertIn("| fake | paraphrase |", table)


if __name__ == "__main__":
    unittest.main()
