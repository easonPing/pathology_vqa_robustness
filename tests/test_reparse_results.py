import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reparse_results.py"
SPEC = importlib.util.spec_from_file_location("reparse_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReparseResultsTest(unittest.TestCase):
    def test_only_invalid_preserves_existing_valid_prediction(self):
        row = {
            "status": "ok",
            "predicted_option": "B",
            "parse_method": "explicit",
            "raw_output": "Final Answer: A",
            "gold_option": "A",
            "A": "Stage I",
            "B": "Stage II",
            "C": "Stage III",
            "D": "Stage IV",
        }
        reparsed = MODULE._reparse_row(row, only_invalid=True)
        self.assertEqual(reparsed["predicted_option"], "B")
        self.assertEqual(reparsed["parse_method"], "explicit")
        self.assertEqual(reparsed["original_predicted_option"], "B")

    def test_only_invalid_recovers_explicit_final_answer(self):
        row = {
            "status": "ok",
            "predicted_option": "INVALID",
            "parse_method": "ambiguous_explicit",
            "raw_output": (
                "Option B is incorrect. Option C is also incorrect.\n"
                "Final Answer: A"
            ),
            "gold_option": "A",
            "A": "Stage I",
            "B": "Stage II",
            "C": "Stage III",
            "D": "Stage IV",
        }
        reparsed = MODULE._reparse_row(row, only_invalid=True)
        self.assertEqual(reparsed["predicted_option"], "A")
        self.assertEqual(reparsed["parse_method"], "explicit")
        self.assertTrue(reparsed["correct"])


if __name__ == "__main__":
    unittest.main()
