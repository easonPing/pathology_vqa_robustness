#!/usr/bin/env bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ROOT="${PATHOLOGY_VQA_ROOT:?}"
BASE_DIR="${PATHOLOGY_VQA_QWEN_TEXT_BASE_DIR:?}"
RERUN_DIR="${PATHOLOGY_VQA_QWEN_TEXT_RERUN_DIR:?}"
ARCHIVE_ROOT="${PATHOLOGY_VQA_ARCHIVE_ROOT:?}"
PYTHON="${CONDA_ROOT:-$HOME/miniconda3}/envs/pathology-vqa-exp2-inference/bin/python"
ARCHIVE_DIR="$ARCHIVE_ROOT/qwen3_text_before_instruction_json_${SLURM_JOB_ID}"

cd "$ROOT"
mkdir -p logs "$ARCHIVE_ROOT"

"$PYTHON" scripts/replace_qwen3_text_instruction_results.py \
    --base-dir "$BASE_DIR" \
    --rerun-dir "$RERUN_DIR" \
    --archive-dir "$ARCHIVE_DIR"
