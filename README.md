# Pathology VQA Perturbation Study

## Setup

Create and activate the environment:

```bash
source ~/.bashrc
conda create -n pathology-vqa-exp2-data-processing python=3.10 -y
conda activate pathology-vqa-exp2-data-processing
PYTHONNOUSERSITE=1 pip install -r data_processing.txt
```

Use `PYTHONNOUSERSITE=1` for project commands on the cluster so packages under
`~/.local` do not override the active Conda environment.

The data-processing environment is for sampling and paraphrase generation.
Create a separate environment for Qwen3-8B, InternVL3-8B, and Qwen3-VL-8B
inference:

```bash
conda create -n pathology-vqa-exp2-inference python=3.10 -y
conda activate pathology-vqa-exp2-inference
PYTHONNOUSERSITE=1 pip install -r inference.txt
PYTHONNOUSERSITE=1 pip install -e . --no-deps
```

SlideChat pins an older Transformers version, so it requires its own environment:

```bash
conda create -n pathology-vqa-exp2-slidechat \
  --override-channels -c conda-forge python=3.10 pip -y
conda activate pathology-vqa-exp2-slidechat
PYTHONNOUSERSITE=1 pip install -r slidechat.txt
PYTHONNOUSERSITE=1 pip install -e . --no-deps
```

Keep model repositories, checkpoints, and Hugging Face caches out of the home
quota:

```bash
export HF_HOME=/bigtemp/efk4ps/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet
```

## Data

The raw benchmark files are stored in:

```text
data/raw/SlideBench-VQA-TCGA.csv
data/raw/SlideBench-VQA-BCNB.csv
```

The frozen Exp2 inputs are stored in:

```text
data/processed/exp2_question_subset.csv
data/processed/vqa_paraphrases.json
data/processed/exp2_question_paraphrased_sample_50.csv
```

The experiment contains 598 questions: 398 TCGA and 200 BCNB. `TCGA-2526`
and `TCGA-2527` were excluded because their standard WSI feature is missing.
The exclusion record is stored in:

```text
data/processed/exp2_exclusions.csv
```

The feature manifest and local feature payloads are stored in:

```text
data/features/survey_prompt_feature/exp2_feature_manifest.csv
data/features/survey_prompt_feature/TCGA_patch_feat/
data/features/survey_prompt_feature/BCNB_patch_feat/
```

Manifest feature paths are repository-relative. Large feature and image payloads
are intentionally ignored by Git.

## Repository Layout

```text
.
|-- data/
|   |-- raw/                         # Original benchmark CSVs
|   |-- processed/                   # Frozen subset and paraphrases
|   |-- features/survey_prompt_feature/
|   `-- images/                      # Native patch manifest; pixels stay local
|-- docs/phase2_patch_acquisition.md
|-- scripts/
|   |-- validate_inputs.py
|   |-- run_experiment.py
|   `-- compute_metrics.py
|-- src/pathology_vqa_exp2/
|   |-- models/                      # Four model adapters
|   |-- data.py, images.py, prompts.py
|   `-- runner.py, answer_parser.py, metrics.py
|-- tests/
|-- stratified_sampling.py
|-- paraphrase_question.py
|-- data_processing.txt
|-- inference.txt
`-- slidechat.txt
```

## Experiment Matrix

Every question expands into eight inference cases:

- four question variants: `canonical`, `p1`, `p2`, and `p3`;
- two bound prompt conditions: `qa` and `instruction`.

These are four logical experiment cells: original, paraphrase, prompt, and
paraphrase+prompt. The three paraphrases remain separate repeated variants, so
the four cells produce eight actual model calls per qid.

The prompt components are not independently toggled:

- `qa` has no system message and directly uses the following format:

```text
Question:
{question}
{non-empty options}
Answer:
```

- `instruction` always uses the exact system message
  `You are an expert pathologist.` together with the model-specific
  instruction format.

The instruction user prompts are:

- SlideChat: `{question}\n{options}`, matching its official test script;
- Qwen3-8B and Qwen3-VL-8B: `{question}\n{options}` followed by the JSON
  option-letter instruction;
- InternVL3-8B: `{question}\n{options}` followed by the official instruction
  `Answer with the option's letter from the given choices directly.`

## Phase 1: SlideChat And Text-Only Qwen

Validate all frozen questions, paraphrases, and feature paths without loading a
model:

```bash
PYTHONNOUSERSITE=1 python scripts/validate_inputs.py
```

Generate the two-question Qwen3-8B smoke-test cases without a GPU:

```bash
PYTHONNOUSERSITE=1 python scripts/run_experiment.py \
  --model qwen3_text \
  --qids TCGA-1 BCNB-31 \
  --output-dir outputs/smoke/paired_prompts/qwen3_text \
  --dry-run
```

Run the same cases on a GPU. Model files are automatically downloaded into
`HF_HOME`; the runner rejects a cache located under the home directory.

```bash
PYTHONNOUSERSITE=1 python scripts/run_experiment.py \
  --model qwen3_text \
  --model-id Qwen/Qwen3-8B \
  --qids TCGA-1 BCNB-31 \
  --output-dir outputs/smoke/paired_prompts/qwen3_text \
  --hf-home /bigtemp/efk4ps/huggingface \
  --dtype bfloat16
```

For SlideChat, accept the access conditions for
`General-Medical-AI/SlideChat_Weight` on Hugging Face and authenticate with
`hf auth login`. Then prepare the pinned official code and stage-2 checkpoint:

```bash
git clone https://github.com/uni-medical/SlideChat.git \
  /bigtemp/efk4ps/pathology_vqa_repos/SlideChat
git -C /bigtemp/efk4ps/pathology_vqa_repos/SlideChat \
  checkout fea92b61f46e0a3786b9674712454bb994c17eb7
PYTHONNOUSERSITE=1 pip install -e \
  /bigtemp/efk4ps/pathology_vqa_repos/SlideChat --no-deps

hf download General-Medical-AI/SlideChat_Weight \
  --revision 98e630bc545c6ecbf9684b053b120d8fb9b7dc8a \
  stage2_pth/mp_rank_00_model_states.pt \
  --local-dir /bigtemp/efk4ps/pathology_vqa_models/SlideChat_Weight
```

Only the 15.3 GB ZeRO-2 model-state file is needed for inference. Do not
download the four stage-2 optimizer shards; together they add about 92 GB and
are only needed to resume training.

Run the SlideChat smoke test using the standard CONCH feature files:

```bash
PYTHONNOUSERSITE=1 python scripts/run_experiment.py \
  --model slidechat \
  --qids TCGA-1 BCNB-31 \
  --slidechat-repo /bigtemp/efk4ps/pathology_vqa_repos/SlideChat \
  --checkpoint /bigtemp/efk4ps/pathology_vqa_models/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt \
  --output-dir outputs/smoke/paired_prompts/slidechat \
  --hf-home /bigtemp/efk4ps/huggingface \
  --dtype float16
```

Each run writes `cases.csv`, an fsync-backed
`predictions.checkpoint.jsonl`, final `predictions.csv`, and
`run_metadata.json`. Resume is enabled by default and skips only successful
cases.

Do not reuse a checkpoint produced by the former 16-case prompt matrix. Its
case IDs and prompt contents are intentionally incompatible with this design.

## Phase 2: Native Patch Images

Phase 2 is currently delegated. The collaborator must recover and freeze 16
raw-pixel patches per unique slide using the existing feature CSV
`patch_name` rows, then write:

```text
data/images/native_patch_manifest.csv
data/images/patch_acquisition_report.json
```

The full acquisition procedure, deterministic sampling algorithm, manifest
schema, and acceptance checks are in
`docs/phase2_patch_acquisition.md`.

## Phase 3: InternVL3 And Qwen3-VL

The native-VLM pipeline can be dry-run before Phase 2 is complete. Each command
writes 16 cases total: eight for one TCGA qid and eight for one BCNB qid.

```bash
PYTHONNOUSERSITE=1 python scripts/run_experiment.py \
  --model internvl3 \
  --qids TCGA-1 BCNB-31 \
  --output-dir outputs/smoke/paired_prompts/internvl3 \
  --dry-run

PYTHONNOUSERSITE=1 python scripts/run_experiment.py \
  --model qwen3_vl \
  --qids TCGA-1 BCNB-31 \
  --output-dir outputs/smoke/paired_prompts/qwen3_vl \
  --dry-run
```

After `native_patch_manifest.csv` passes Phase 2 acceptance, run the GPU smoke
tests. All model downloads and Hugging Face caches stay under `/bigtemp`.

```bash
PYTHONNOUSERSITE=1 python scripts/run_experiment.py \
  --model internvl3 \
  --model-id OpenGVLab/InternVL3-8B \
  --qids TCGA-1 BCNB-31 \
  --image-manifest data/images/native_patch_manifest.csv \
  --images-per-question 16 \
  --output-dir outputs/smoke/paired_prompts/internvl3 \
  --hf-home /bigtemp/efk4ps/huggingface \
  --dtype bfloat16

PYTHONNOUSERSITE=1 python scripts/run_experiment.py \
  --model qwen3_vl \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --qids TCGA-1 BCNB-31 \
  --image-manifest data/images/native_patch_manifest.csv \
  --images-per-question 16 \
  --output-dir outputs/smoke/paired_prompts/qwen3_vl \
  --hf-home /bigtemp/efk4ps/huggingface \
  --dtype bfloat16
```

For the full 598-question run, replace `--qids TCGA-1 BCNB-31` with `--all`
and use a new output directory. InternVL resizes each already-frozen patch to
448 x 448 without dynamic tiling. Qwen3-VL uses its official processor and the
same manifest image order.

## Metrics

Compute the three requested perturbation groups separately for each model:

```bash
PYTHONNOUSERSITE=1 python scripts/compute_metrics.py \
  --predictions \
    outputs/smoke/paired_prompts/qwen3_text/predictions.csv \
    outputs/smoke/paired_prompts/slidechat/predictions.csv \
  --output-dir outputs/smoke/paired_prompts/metrics
```

The groups are `paraphrase`, `prompt`, and `paraphrase+prompt`, each anchored to
`canonical x qa`. Their perturbed rows are respectively `p1/p2/p3 x qa`,
`canonical x instruction`, and `p1/p2/p3 x instruction`. The command reports
canonical and perturbed accuracy, their difference, question-level bootstrap
95% confidence intervals, consistency, pairwise flip-rate, base-2 Shannon
entropy, and selected-option distribution. It writes
`per_model_perturbation_metrics.csv`,
`per_question_metrics.csv`, `position_bias.csv`, and `exp2_table4_row.md`.

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
  --output data/processed/vqa_paraphrases.json \
  --audit-output data/processed/exp2_question_paraphrased_sample_50.csv \
  --temperature 0.7
```

## Outputs

`data/processed/vqa_paraphrases.json` contains all accepted paraphrases:

```json
{
  "metadata": {
    "source_subset": "data/processed/exp2_question_subset.csv",
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

`data/processed/exp2_question_paraphrased_sample_50.csv` contains a stratified
50-question human audit sample with the original question, options, answer, and
three paraphrases.
