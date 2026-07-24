import unittest
from pathlib import Path

from pathology_vqa_exp2.data import build_experiment_cases, load_questions
from pathology_vqa_exp2.prompts import (
    EXPERT_SYSTEM_PROMPT,
    INTERNVL_MCQ_INSTRUCTION,
    QA_JSON_ANSWER_INSTRUCTION,
    QWEN3_TEXT_MCQ_INSTRUCTION,
    QWEN3_VL_MCQ_INSTRUCTION,
    render_prompt,
)


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
        self.assertIn(QWEN3_TEXT_MCQ_INSTRUCTION, instruction.user_prompt)
        self.assertIn(QA_JSON_ANSWER_INSTRUCTION, instruction.user_prompt)
        self.assertTrue(instruction.user_prompt.endswith("\nAnswer:"))
        self.assertEqual(instruction.system_prompt, EXPERT_SYSTEM_PROMPT)
        self.assertTrue(qa.user_prompt.startswith("Question:\n"))
        self.assertTrue(qa.user_prompt.endswith("\nAnswer:"))
        self.assertIn(QA_JSON_ANSWER_INSTRUCTION, qa.user_prompt)
        self.assertIsNone(qa.system_prompt)

    def test_all_model_prompt_templates_are_exact(self):
        question = "Which diagnosis is most likely?"
        options = {"A": "Alpha", "B": "Beta", "C": "", "D": "Delta"}
        options_text = "A. Alpha\nB. Beta\nD. Delta"
        expected_qa = (
            f"Question:\n{question}\n{options_text}\n"
            f"{QA_JSON_ANSWER_INSTRUCTION}\nAnswer:"
        )
        expected_instruction = {
            "slidechat": f"{question}\n{options_text}",
            "qwen3_text": (
                f"{question}\n{options_text}\n{QWEN3_TEXT_MCQ_INSTRUCTION}\n"
                f"{QA_JSON_ANSWER_INSTRUCTION}\nAnswer:"
            ),
            "qwen3_vl": (
                f"Question: {question}\nOptions:\n{options_text}\n"
                f"{QWEN3_VL_MCQ_INSTRUCTION}\n"
                f"{QA_JSON_ANSWER_INSTRUCTION}\nAnswer:"
            ),
            "internvl3": (
                f"{question}\n{options_text}\n{INTERNVL_MCQ_INSTRUCTION}"
            ),
        }

        for model_key, expected_user_prompt in expected_instruction.items():
            with self.subTest(model=model_key, condition="qa"):
                system_prompt, user_prompt = render_prompt(
                    model_key, question, options, "qa"
                )
                self.assertIsNone(system_prompt)
                self.assertEqual(user_prompt, expected_qa)
            with self.subTest(model=model_key, condition="instruction"):
                system_prompt, user_prompt = render_prompt(
                    model_key, question, options, "instruction"
                )
                self.assertEqual(system_prompt, EXPERT_SYSTEM_PROMPT)
                self.assertEqual(user_prompt, expected_user_prompt)


if __name__ == "__main__":
    unittest.main()
