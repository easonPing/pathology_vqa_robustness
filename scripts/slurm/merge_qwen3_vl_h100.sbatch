#!/usr/bin/env bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ROOT="${PATHOLOGY_VQA_ROOT:-${SLURM_SUBMIT_DIR:?}}"
DATA_ROOT="${PATHOLOGY_VQA_DATA_ROOT:?}"
RESULTS_ROOT="${PATHOLOGY_VQA_RESULTS_ROOT:?}"
BASE_JOB_ID="${PATHOLOGY_VQA_BASE_JOB_ID:?}"
PYTHON="$HOME/miniconda3/envs/pathology-vqa-exp2-inference/bin/python"
BASE_CHECKPOINT="$RESULTS_ROOT/full/qwen3_vl/patch_predictions.checkpoint.jsonl"
TARGET_BASE_PATCH_ROWS=$((251 * 8 * 30))

cd "$ROOT"
mkdir -p logs "$RESULTS_ROOT/full/qwen3_vl_merged"

while true; do
    completed_rows=0
    if [[ -f "$BASE_CHECKPOINT" ]]; then
        completed_rows="$(wc -l < "$BASE_CHECKPOINT")"
    fi
    if (( completed_rows >= TARGET_BASE_PATCH_ROWS )); then
        break
    fi
    if ! squeue -h -j "$BASE_JOB_ID" | grep -q .; then
        echo "Base H100 job ended with only $completed_rows/$TARGET_BASE_PATCH_ROWS rows" >&2
        exit 1
    fi
    echo "waiting for base checkpoint: $completed_rows/$TARGET_BASE_PATCH_ROWS"
    sleep 60
done

"$PYTHON" scripts/merge_qwen3_vl_shards.py \
    --subset "$DATA_ROOT/processed/exp2_question_subset.csv" \
    --paraphrases "$DATA_ROOT/processed/vqa_paraphrases.json" \
    --image-manifest "$DATA_ROOT/manifests/native_patch_manifest.csv" \
    --output-dir "$RESULTS_ROOT/full/qwen3_vl_merged" \
    --source 0 251 "$BASE_CHECKPOINT" \
    --source 251 294 "$RESULTS_ROOT/full/qwen3_vl/questions_0251_0294/patch_predictions.checkpoint.jsonl" \
    --source 294 469 "$RESULTS_ROOT/full/qwen3_vl/questions_0294_0469/patch_predictions.checkpoint.jsonl" \
    --source 469 598 "$RESULTS_ROOT/full/qwen3_vl/questions_0469_0598/patch_predictions.checkpoint.jsonl"

# The merged output is complete and validated, so the unsharded base job can
# stop instead of recomputing ranges covered by the other H100 jobs.
scancel "$BASE_JOB_ID" || true
