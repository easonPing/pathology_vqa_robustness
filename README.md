# pathology-vqa-perturbation-study

## Setup

Create and activate the environment:

```bash
source ~/.bashrc
conda create -n pathology-vqa-exp2 python=3.10 -y
conda activate pathology-vqa-exp2
pip install -r requirements.txt
```

## Data

The raw benchmark files are stored in:

```text
data/raw/SlideBench-VQA-TCGA.csv
data/raw/SlideBench-VQA-BCNB.csv
```

The fixed Exp2 question subset is stored in:

```text
data/exp2_question_subset.csv
```

Validate and prepare the fixed subset:

```bash
python stratified_sampling.py
```

This verifies the 400 TCGA / 200 BCNB proportional task distribution and rewrites
`data/exp2_question_subset.csv` with the canonical columns.

## Paraphrase Generation

Run a small local smoke test with Qwen3-1.7B:

```bash
python paraphrase_question.py \
  --model-id Qwen/Qwen3-1.7B \
  --checker-model-id Qwen/Qwen3-1.7B \
  --limit 1 \
  --output /tmp/vqa_paraphrases_smoke.json \
  --audit-output /tmp/exp2_question_paraphrased_sample_50_smoke.csv \
  --checkpoint /tmp/vqa_paraphrases_smoke.checkpoint.json \
  --temperature 0.7
```

Run the final paraphrase generation on a GPU node with Qwen3-32B:

```bash
python paraphrase_question.py \
  --model-id Qwen/Qwen3-32B \
  --checker-model-id Qwen/Qwen3-32B \
  --output vqa_paraphrases.json \
  --audit-output exp2_question_paraphrased_sample_50.csv \
  --temperature 0.7
```

## Outputs

`vqa_paraphrases.json` contains all accepted paraphrases:

```json
{
  "metadata": {
    "source_subset": "data/exp2_question_subset.csv",
    "generator_model": "Qwen/Qwen3-32B",
    "checker_model": "Qwen/Qwen3-32B",
    "k": 3,
    "seed": 42
  },
  "items": [
    {
      "qid": "TCGA-1",
      "dataset": "TCGA",
      "ID": "1",
      "Slide": "TCGA-...",
      "Task": "Staging",
      "Question": "Original question text",
      "options": {
        "A": "option A",
        "B": "option B",
        "C": "option C",
        "D": "option D"
      },
      "non_empty_options": ["A", "B", "C", "D"],
      "Answer": "C",
      "paraphrases": [
        {
          "paraphrase_id": "TCGA-1-p1",
          "Question": "Accepted paraphrased question"
        }
      ]
    }
  ]
}
```

`exp2_question_paraphrased_sample_50.csv` contains a stratified 50-question human
audit sample with the original question, options, answer, and three paraphrases.
