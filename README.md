# Pathology VQA Prompt-Robustness Study

This repository runs the frozen SlideBench-VQA prompt-robustness experiment
requested for the survey paper. Large data, model weights, caches, and results
live inside this repository tree but are excluded from Git.

## Frozen experimental scope

- Questions: 598 total (398 TCGA and 200 BCNB).
- Exclusions: `TCGA-2526` and `TCGA-2527`, approved because the shared WSI
  feature is unavailable.
- Question variants: canonical plus three approved paraphrases (`p1`–`p3`).
- Prompt conditions: exactly two bound conditions:
  - `qa`: no system message and `Question: ... Answer:` formatting;
  - `instruction`: `You are an expert pathologist.` plus the prompt format
    specific to each model.
- Models: SlideChat, Qwen3-8B text-only, Qwen3-VL-8B, and InternVL3-8B.
- Option shuffling is not part of this repository's assigned experiment; the
  supervisor is handling it separately.

Each question therefore has 4 question variants x 2 prompt conditions = 8
slide-level cases per model. The 598-question subset and its paraphrases are
frozen inputs and must not be regenerated or edited.

For each native VLM (`qwen3_vl` and `internvl3`), every slide-level case is
expanded into 30 independent, deterministic, single-patch generations. Thus a
full native-model run has 598 x 8 x 30 = 143,520 patch calls. The same ordered
30 patches are reused for all questions on the same slide, all variants, both
prompt conditions, and both native VLMs.

## Repository layout

```text
.
|-- configs/
|   |-- models_config/models.json       # model IDs and repo-local paths
|   `-- native_patch_protocol.json      # frozen 30-patch/voting protocol
|-- data/                               # ignored by Git
|   |-- raw/
|   |   |-- benchmarks/                 # SlideBench TCGA/BCNB CSVs
|   |   `-- slides/{TCGA,BCNB}/         # raw SVS/JPG/JSON payloads
|   |-- processed/                      # frozen 598-question inputs
|   |-- features/slidechat/             # delivered SlideChat feature CSVs
|   |-- patches/native_vlm/             # frozen 224x224 native-VLM patches
|   `-- manifests/                      # selections, hashes, and path maps
|-- models/                             # ignored weights and HF cache
|   |-- huggingface/
|   `-- SlideChat_Weight/
|-- third_party/SlideChat/              # pinned official repository, ignored
|-- scripts/
|   |-- data/
|   |   |-- select_native_patches.py
|   |   |-- extract_native_patches.py
|   |   `-- build_native_patch_manifest.py
|   |-- validate_inputs.py
|   |-- run_experiment.py
|   `-- compute_metrics.py
|-- src/pathology_vqa_exp2/
|-- tests/
`-- results/                            # generated runs, ignored by Git
```

`docs/` and `outputs/` are intentionally absent. The required instructions
are maintained here, and new runs go to `results/`.

Canonical local paths are:

```text
data/raw/benchmarks/SlideBench-VQA-TCGA.csv
data/raw/benchmarks/SlideBench-VQA-BCNB.csv
data/processed/exp2_question_subset.csv
data/processed/vqa_paraphrases.json
data/processed/exp2_question_paraphrased_sample_50.csv
data/processed/exp2_exclusions.csv
data/manifests/slidechat_feature_manifest.csv
data/features/slidechat/TCGA_patch_feat/
data/features/slidechat/BCNB_patch_feat/
```

Because all of `data/` is ignored, keep a separate backup of the small frozen
files under `data/processed/` and `data/manifests/`.

## Environment setup

Run commands from the repository root. Model caches are repo-local by default:

```bash
export HF_HOME="$PWD/models/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
```

Data preparation:

```bash
conda create -n pathology-vqa-exp2-data-processing python=3.10 -y
conda activate pathology-vqa-exp2-data-processing
PYTHONNOUSERSITE=1 pip install -r data_processing.txt
PYTHONNOUSERSITE=1 pip install -e . --no-deps
```

Patch reconstruction also needs the system OpenSlide and libvips libraries.
Install them through the cluster package/module system or Conda before using
`extract_native_patches.py`.

Qwen3/InternVL inference:

```bash
conda create -n pathology-vqa-exp2-inference python=3.10 -y
conda activate pathology-vqa-exp2-inference
PYTHONNOUSERSITE=1 pip install -r inference.txt
PYTHONNOUSERSITE=1 pip install -e . --no-deps
```

SlideChat uses an older Transformers stack and must remain separate:

```bash
conda create -n pathology-vqa-exp2-slidechat \
  --override-channels -c conda-forge python=3.10 pip -y
conda activate pathology-vqa-exp2-slidechat
PYTHONNOUSERSITE=1 pip install -r slidechat.txt
PYTHONNOUSERSITE=1 pip install -e . --no-deps
```

Install the pinned SlideChat source and checkpoint at the paths recorded in
`configs/models_config/models.json`:

```bash
git clone https://github.com/uni-medical/SlideChat.git third_party/SlideChat
git -C third_party/SlideChat checkout \
  fea92b61f46e0a3786b9674712454bb994c17eb7
PYTHONNOUSERSITE=1 pip install -e third_party/SlideChat --no-deps

hf download General-Medical-AI/SlideChat_Weight \
  --revision 98e630bc545c6ecbf9684b053b120d8fb9b7dc8a \
  --local-dir models/SlideChat_Weight
```

## Native patch preparation

The preferred path is to copy the original delivered patches plus their golden
SHA256 validation CSV. Reconstruct from raw slides only when necessary.

1. Freeze the shared 30-patch selection from the delivered SlideChat feature
   pools:

```bash
PYTHONNOUSERSITE=1 python scripts/data/select_native_patches.py
```

This writes `data/manifests/native_patch_selection.csv`. The selection is
without replacement and uses a stable per-slide seed. The stable seed is a
project reproducibility choice; SlideChat's paper does not publish its seed or
RNG implementation.

2. If original patches are unavailable, reconstruct and require a full golden
   checksum gate:

```bash
PYTHONNOUSERSITE=1 python scripts/data/extract_native_patches.py \
  --reference-checksums data/manifests/harry_patch_validation.csv
```

This writes patches under `data/patches/native_vlm/` and a slide-level report
at `data/manifests/native_patch_extraction_report.csv`. Without golden hashes,
the command refuses production extraction; the explicit
`--allow-unverified-reconstruction` flag is for exploration only.

3. Expand the verified 407-slide report into the canonical 598-question
   manifest:

```bash
PYTHONNOUSERSITE=1 python scripts/data/build_native_patch_manifest.py
```

By default this reopens, validates, and hashes all 12,210 unique slide patches,
then writes 17,940 qid-level rows to
`data/manifests/native_patch_manifest.csv`.

### Reconstruction limitation

SlideChat publicly specifies random selection of 30 patches followed by a
slide-level vote, but its repository consumes 512-dimensional feature CSVs and
does not publish the TCGA/BCNB `patch_name -> source pixels` inverse mapping.

Harry's BCNB reconstruction follows the public BALNMP naming and broad
polygon/bounding-box/256-grid structure, but it is not pixel-equivalent to the
official preprocessing. BALNMP uses OpenCV polygon rasterization, an
intermediate quality-100 JPEG, a blank-ratio filter, and quality-100 patch
JPEGs; Harry's path uses pyvips/PIL, directly reconstructs, resizes to 224, and
writes quality 95. The TCGA 512-pixel coordinate interpretation is plausible
from the delivered manifest but is not documented by SlideChat. Publication
runs must therefore use original patches or pass the golden checksum gate.

Official sources: [SlideChat paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_SlideChat_A_Large_Vision-Language_Assistant_for_Whole-Slide_Pathology_Image_Understanding_CVPR_2025_paper.pdf),
[SlideChat repository](https://github.com/uni-medical/SlideChat),
[BALNMP tumor-region preprocessing](https://github.com/bupt-ai-cz/BALNMP/blob/main/preprocessing/cut_tumor_regions.py),
and [BALNMP patch preprocessing](https://github.com/bupt-ai-cz/BALNMP/blob/main/preprocessing/cut_patches.py).

## Prompt conditions

The two prompt components are paired; there are no independent system/format
toggles.

`qa`:

```text
Question:
{question}
{non-empty options}
Answer:
```

It has no system message.

`instruction` always has this system message:

```text
You are an expert pathologist.
```

The user message remains model-specific:

- SlideChat: question plus options, matching its released test input shape;
- Qwen3 text/VL: question plus options and a strict JSON answer instruction;
- InternVL3: question plus options and its direct option-letter instruction.

## Validation and inference

Validate all 598 frozen records and feature-manifest alignment without requiring
the large feature payload:

```bash
PYTHONNOUSERSITE=1 python scripts/validate_inputs.py
```

After feature transfer, require all referenced feature CSVs:

```bash
PYTHONNOUSERSITE=1 python scripts/validate_inputs.py --require-feature-files
```

Text-only smoke test:

```bash
PYTHONNOUSERSITE=1 python scripts/run_experiment.py \
  --model qwen3_text \
  --qids TCGA-1 BCNB-31 \
  --output-dir results/smoke/qwen3_text \
  --dry-run
```

SlideChat smoke test:

```bash
PYTHONNOUSERSITE=1 python scripts/run_experiment.py \
  --model slidechat \
  --qids TCGA-1 BCNB-31 \
  --output-dir results/smoke/slidechat
```

Native VLM smoke tests use the canonical manifest and strictly require 30
verified image files per question:

```bash
PYTHONNOUSERSITE=1 python scripts/run_experiment.py \
  --model qwen3_vl \
  --qids TCGA-1 BCNB-31 \
  --output-dir results/smoke/qwen3_vl \
  --dry-run

PYTHONNOUSERSITE=1 python scripts/run_experiment.py \
  --model internvl3 \
  --qids TCGA-1 BCNB-31 \
  --output-dir results/smoke/internvl3 \
  --dry-run
```

Remove `--dry-run` to load the model and execute inference. Use `--all` instead
of `--qids ...` for the complete frozen set.

For native VLMs, each run directory contains:

```text
cases.csv                              # 8 slide-level cases per qid
patch_cases.csv                        # 30 calls per slide-level case
patch_predictions.checkpoint.jsonl     # resumable append-only raw records
patch_predictions.csv                  # all 30 raw outputs and parsed votes
predictions.csv                        # slide-level aggregated results
run_metadata.json                      # model revision and frozen protocol
```

## Voting and non-A/B/C/D outputs

The SlideChat paper says “majority voting” but neither the paper nor released
runner defines tie-breaking, invalid-output handling, retries, or strict-majority
versus plurality. The repository's MMBench helper contains a permissive A-D
regex, but that is not a published SlideBench 30-patch protocol and can match
letters inside prose.

Until the supervisor confirms the authors' hidden implementation, this project
uses the conservative policy frozen in `configs/native_patch_protocol.json`:

1. Parse each raw response deterministically; otherwise label it `INVALID`.
2. Count `A`, `B`, `C`, `D`, and `INVALID` as five vote categories.
3. Accept a valid answer only when it is the unique highest-count category.
4. If `INVALID` wins, the maximum is tied, or no valid vote exists, emit
   slide-level `INVALID`.
5. Never retry, sample a 31st patch, use model confidence, or break a tie by
   letter/patch order.

`predictions.csv` records all vote counts and `vote_status`; accuracy counts
`INVALID` and ties as incorrect. Metrics also report invalid and tie rates.

## Metrics

After all model runs finish:

```bash
PYTHONNOUSERSITE=1 python scripts/compute_metrics.py \
  --predictions \
    results/full/qwen3_text/predictions.csv \
    results/full/slidechat/predictions.csv \
    results/full/qwen3_vl/predictions.csv \
    results/full/internvl3/predictions.csv \
  --output-dir results/full/metrics
```

Outputs include canonical and perturbed accuracy, question-level bootstrap 95%
confidence intervals, accuracy delta, consistency, pairwise flip rate, Shannon
entropy, invalid/tie rates, per-question rows, and selected-option
distributions. The selected-option distribution is descriptive only here; a
causal position-bias analysis requires the separately assigned option-shuffle
experiment.
