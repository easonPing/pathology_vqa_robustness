#!/usr/bin/env bash
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

TAIL_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this file with --array=4-7}"
if (( TAIL_SHARD_INDEX < 4 || TAIL_SHARD_INDEX > 7 )); then
    echo "SLURM_ARRAY_TASK_ID must be one of 4, 5, 6, or 7" >&2
    exit 2
fi

export PATHOLOGY_VQA_NUM_SHARDS=8
export PATHOLOGY_VQA_SHARD_INDEX="$TAIL_SHARD_INDEX"

# The shared model launcher uses array task 1 for Qwen3-VL. Preserve the
# outer array index above as the question-shard index before dispatching.
export SLURM_ARRAY_TASK_ID=1
exec bash scripts/slurm/run_model_array.sbatch full
