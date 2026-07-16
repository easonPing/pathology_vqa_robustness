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
