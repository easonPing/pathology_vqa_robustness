import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JSON_INSTRUCTION = (
    "Return your choice as JSON with only the option letter in the answer "
    'field, for example: {"answer": "C"}.'
)
ADAPTER = {
    "model_key": "qwen3_text",
    "model_id": "Qwen/Qwen3-8B",
    "revision": "test-revision",
    "dtype": "bfloat16",
    "max_new_tokens": 512,
    "thinking": False,
    "do_sample": False,
}


class ReplaceQwenInstructionTest(unittest.TestCase):
    @staticmethod
    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def write_run(directory: Path, cases: list[dict], predictions: list[dict]) -> None:
        directory.mkdir(parents=True)
        ReplaceQwenInstructionTest.write_csv(directory / "cases.csv", cases)
        ReplaceQwenInstructionTest.write_csv(
            directory / "predictions.csv", predictions
        )
        with (directory / "predictions.checkpoint.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in predictions:
                handle.write(json.dumps(row) + "\n")
        (directory / "run_metadata.json").write_text(
            json.dumps(
                {
                    "adapter": ADAPTER,
                    "started_at": "old-start",
                    "finished_at": "new-finish",
                }
            ),
            encoding="utf-8",
        )

    def test_replaces_only_instruction_and_archives_original(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            base = root / "base"
            rerun = root / "rerun"
            archive = root / "archive"
            qa_case = {
                "case_id": "Q1::canonical::qa",
                "model_key": "qwen3_text",
                "prompt_condition": "qa",
                "system_prompt": "",
                "user_prompt": "Question:\nQ?\nAnswer:",
                "feature_path": "",
            }
            old_instruction_case = {
                "case_id": "Q1::canonical::instruction",
                "model_key": "qwen3_text",
                "prompt_condition": "instruction",
                "system_prompt": "You are an expert pathologist.",
                "user_prompt": "old instruction",
                "feature_path": "",
            }
            new_instruction_case = {
                **old_instruction_case,
                "user_prompt": (
                    "Q?\nA. Alpha\nB. Beta\n"
                    f"{JSON_INSTRUCTION}\nAnswer:"
                ),
            }
            old_predictions = [
                {
                    **qa_case,
                    "raw_output": '{"answer": "A"}',
                    "predicted_option": "A",
                    "correct": True,
                    "status": "ok",
                },
                {
                    **old_instruction_case,
                    "raw_output": "cannot answer",
                    "predicted_option": "INVALID",
                    "correct": False,
                    "status": "ok",
                },
            ]
            new_prediction = {
                **new_instruction_case,
                "raw_output": '{"answer": "B"}',
                "predicted_option": "B",
                "correct": True,
                "status": "ok",
            }
            self.write_run(
                base, [qa_case, old_instruction_case], old_predictions
            )
            self.write_run(rerun, [new_instruction_case], [new_prediction])

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/replace_qwen3_text_instruction_results.py"),
                    "--base-dir",
                    str(base),
                    "--rerun-dir",
                    str(rerun),
                    "--archive-dir",
                    str(archive),
                    "--expected-per-condition",
                    "1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (base / "predictions.checkpoint.jsonl").open(
                encoding="utf-8"
            ) as handle:
                merged = [json.loads(line) for line in handle]
            self.assertEqual(merged[0]["raw_output"], '{"answer": "A"}')
            self.assertEqual(merged[1]["raw_output"], '{"answer": "B"}')
            self.assertEqual(
                json.loads(
                    (base / "instruction_replacement.json").read_text()
                )["old_instruction_invalid"],
                1,
            )
            self.assertTrue((archive / "predictions.csv").is_file())
            with (archive / "predictions.checkpoint.jsonl").open(
                encoding="utf-8"
            ) as handle:
                archived = [json.loads(line) for line in handle]
            self.assertEqual(archived[1]["predicted_option"], "INVALID")


if __name__ == "__main__":
    unittest.main()
