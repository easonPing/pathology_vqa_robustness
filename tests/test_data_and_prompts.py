import unittest
from pathlib import Path

from pathology_vqa_exp2.data import build_experiment_cases, load_questions
from pathology_vqa_exp2.prompts import EXPERT_SYSTEM_PROMPT, QWEN_MCQ_INSTRUCTION


ROOT = Path(__file__).resolve().parents[1]


class DataAndPromptTest(unittest.TestCase):
    def load_smoke_records(self, require_features=False):
        return load_questions(
            ROOT / "data/processed/exp2_question_subset.csv",
            ROOT / "data/processed/vqa_paraphrases.json",
            feature_manifest_path=ROOT
            / "data/manifests/slidechat_feature_manifest.csv",
            require_features=require_features,
            qids=["TCGA-1", "BCNB-31"],
        )

    def test_smoke_inputs_are_aligned_and_reference_features(self):
        records = self.load_smoke_records(require_features=False)
        self.assertEqual([record.qid for record in records], ["TCGA-1", "BCNB-31"])
        self.assertTrue(all(record.feature_path is not None for record in records))
        self.assertTrue(
            all(
                "data/features/slidechat" in record.feature_path.as_posix()
                for record in records
            )
        )
        self.assertEqual(records[0].valid_options, ("A", "B", "C", "D"))
        self.assertEqual(records[1].valid_options, ("A", "B", "C"))

    def test_each_question_expands_to_8_cases(self):
        records = self.load_smoke_records()
        cases = build_experiment_cases(records, "qwen3_text")
        self.assertEqual(len(cases), 16)
        self.assertEqual(len({case.case_id for case in cases}), 16)

    def test_qwen_instruction_and_qa_prompts_are_bound(self):
        record = self.load_smoke_records()[0]
        cases = build_experiment_cases([record], "qwen3_text")
        instruction = next(
            case
            for case in cases
            if case.question_variant == "canonical"
            and case.prompt_condition == "instruction"
        )
        qa = next(
            case
            for case in cases
            if case.question_variant == "canonical"
            and case.prompt_condition == "qa"
        )
        self.assertIn(QWEN_MCQ_INSTRUCTION, instruction.user_prompt)
        self.assertEqual(instruction.system_prompt, EXPERT_SYSTEM_PROMPT)
        self.assertTrue(qa.user_prompt.startswith("Question:\n"))
        self.assertTrue(qa.user_prompt.endswith("\nAnswer:"))
        self.assertIsNone(qa.system_prompt)

    def test_slidechat_native_matches_official_question_options_shape(self):
        record = self.load_smoke_records()[0]
        cases = build_experiment_cases([record], "slidechat")
        instruction = next(
            case
            for case in cases
            if case.question_variant == "canonical"
            and case.prompt_condition == "instruction"
        )
        self.assertNotIn(QWEN_MCQ_INSTRUCTION, instruction.user_prompt)
        self.assertEqual(instruction.user_prompt.splitlines()[1], "A. Stage I")
        self.assertEqual(instruction.system_prompt, EXPERT_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
