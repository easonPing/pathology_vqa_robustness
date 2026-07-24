import unittest

from pathology_vqa_exp2.answer_parser import INVALID_OPTION, parse_answer


OPTIONS = {
    "A": "Stage I",
    "B": "Stage II",
    "C": "Stage III",
    "D": "Stage IV",
}


class AnswerParserTest(unittest.TestCase):
    def test_json_answer(self):
        parsed = parse_answer('{"answer": "C"}', OPTIONS)
        self.assertEqual((parsed.option, parsed.method), ("C", "json"))

    def test_fenced_json_after_thinking(self):
        parsed = parse_answer(
            '<think>Reasoning mentions A and B.</think>\n```json\n{"answer":"D"}\n```',
            OPTIONS,
        )
        self.assertEqual(parsed.option, "D")

    def test_explicit_answer(self):
        self.assertEqual(parse_answer("The final answer is B.", OPTIONS).option, "B")

    def test_unwrapped_json_field(self):
        parsed = parse_answer('"answer": "A"', OPTIONS)
        self.assertEqual((parsed.option, parsed.method), ("A", "explicit"))

    def test_truncated_json_after_reasoning(self):
        parsed = parse_answer(
            'The evidence supports the first option.\n```json\n{"answer": "A',
            OPTIONS,
        )
        self.assertEqual((parsed.option, parsed.method), ("A", "explicit"))

    def test_markdown_answer_label(self):
        parsed = parse_answer("The tumor is high grade.\n\n**Answer:** D", OPTIONS)
        self.assertEqual((parsed.option, parsed.method), ("D", "explicit"))

    def test_markdown_option_after_correct_answer_label(self):
        parsed = parse_answer(
            "Therefore, the correct answer is:\n\n**C. Stage III**", OPTIONS
        )
        self.assertEqual((parsed.option, parsed.method), ("C", "explicit"))

    def test_final_answer_wins_over_rejected_option_mentions(self):
        parsed = parse_answer(
            "Option B describes a benign lesion and is incorrect. "
            "Option C is also incorrect.\n\n"
            "✅ Final Answer: **A. Stage I**",
            OPTIONS,
        )
        self.assertEqual((parsed.option, parsed.method), ("A", "explicit"))

    def test_parenthetical_final_answer_label(self):
        parsed = parse_answer(
            "After reviewing the evidence:\n"
            "**Final Answer (based on the available information):** "
            "**D. Stage IV**",
            OPTIONS,
        )
        self.assertEqual((parsed.option, parsed.method), ("D", "explicit"))

    def test_non_option_final_answer_does_not_recover_speculative_letter(self):
        output = (
            "If forced to guess, Option A might be selected.\n"
            "Final Answer: None of the options can be determined."
        )
        parsed = parse_answer(output, OPTIONS)
        self.assertEqual(
            (parsed.option, parsed.method),
            (INVALID_OPTION, "non_option_final_answer"),
        )

    def test_conflicting_final_answers_are_invalid(self):
        output = "Final Answer: A\nCorrection — Final Answer: C"
        self.assertEqual(parse_answer(output, OPTIONS).option, INVALID_OPTION)

    def test_answer_field_with_conflicting_letters_is_invalid(self):
        output = '"answer": "A"\nCorrection: "answer": "C"'
        self.assertEqual(parse_answer(output, OPTIONS).option, INVALID_OPTION)

    def test_truncated_reasoning_without_answer_is_invalid(self):
        # Do not mistake the trailing English article "a" for option A.
        output = "Without the complete slide, it is not possible to assign a"
        self.assertEqual(parse_answer(output, OPTIONS).option, INVALID_OPTION)

    def test_exact_option_text(self):
        self.assertEqual(parse_answer("Stage III", OPTIONS).option, "C")

    def test_leading_letter_with_option_text(self):
        parsed = parse_answer("C. Stage III", OPTIONS)
        self.assertEqual((parsed.option, parsed.method), ("C", "leading_letter"))

    def test_missing_option_is_invalid(self):
        three_options = {**OPTIONS, "D": ""}
        self.assertEqual(parse_answer('{"answer":"D"}', three_options).option, INVALID_OPTION)

    def test_conflicting_json_is_invalid(self):
        output = '{"answer":"A"}\n{"answer":"C"}'
        self.assertEqual(parse_answer(output, OPTIONS).option, INVALID_OPTION)


if __name__ == "__main__":
    unittest.main()
