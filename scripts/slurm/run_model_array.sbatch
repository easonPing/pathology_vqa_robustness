#!/usr/bin/env bash
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "test" && "$MODE" != "full" ]]; then
    echo "Usage: sbatch [Slurm options] $0 {test|full}" >&2
    exit 2
fi

case "${SLURM_ARRAY_TASK_ID:?Submit this file with --array=0-3}" in
    0)
        MODEL="qwen3_text"
        ENVIRONMENT="pathology-vqa-exp2-inference"
        ;;
    1)
        MODEL="qwen3_vl"
        ENVIRONMENT="pathology-vqa-exp2-inference"
        ;;
    2)
        MODEL="internvl3"
        ENVIRONMENT="pathology-vqa-exp2-inference"
        ;;
    3)
        MODEL="slidechat"
        ENVIRONMENT="pathology-vqa-exp2-slidechat"
        ;;
    *)
        echo "SLURM_ARRAY_TASK_ID must be one of 0, 1, 2, or 3" >&2
        exit 2
        ;;
esac

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Slurm executes a spool copy of this script, so BASH_SOURCE no longer points
# into the repository. Prefer an explicit override, then the submit directory.
ROOT="${PATHOLOGY_VQA_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_ROOT}}"
ROOT="$(cd "$ROOT" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
PYTHON="$CONDA_ROOT/envs/$ENVIRONMENT/bin/python"
BIGTEMP_USER="${USER:-$(id -un)}"
MODEL_ROOT="${PATHOLOGY_VQA_MODEL_ROOT:-/bigtemp/$BIGTEMP_USER/pathology_vqa_robustness/models}"
HF_HOME="$MODEL_ROOT/huggingface"
SLIDECHAT_CHECKPOINT="$MODEL_ROOT/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt"
TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/pathology-vqa-triton-${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}"
EFFECTIVE_MODEL_DTYPE="${MODEL_DTYPE:-}"
if [[ "$MODEL" == "internvl3" && -n "${INTERNVL_DTYPE:-}" ]]; then
    EFFECTIVE_MODEL_DTYPE="$INTERNVL_DTYPE"
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Python environment: $PYTHON" >&2
    exit 1
fi

required_inputs=(
    "$ROOT/data/processed/exp2_question_subset.csv"
    "$ROOT/data/processed/vqa_paraphrases.json"
    "$ROOT/data/manifests/slidechat_feature_manifest.csv"
    "$ROOT/data/manifests/native_patch_manifest.csv"
)
for path in "${required_inputs[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required input: $path" >&2
        exit 1
    fi
done

if [[ "$MODEL" == "slidechat" ]]; then
    module load cuda/12.8.1
    # The SlideChat environment uses PyTorch cu121, while the cluster's
    # available toolkit is newer. DeepSpeed ops are not compiled or used by
    # this inference path, so permit the runtime-only version mismatch.
    export DS_SKIP_CUDA_CHECK=1
    slidechat_source="$ROOT/third_party/SlideChat/xtuner/model/llava.py"
    if [[ ! -f "$slidechat_source" ]]; then
        echo "Missing pinned SlideChat source: $slidechat_source" >&2
        exit 1
    fi
    if [[ ! -f "$SLIDECHAT_CHECKPOINT" ]]; then
        echo "Missing SlideChat checkpoint: $SLIDECHAT_CHECKPOINT" >&2
        exit 1
    fi
fi

cd "$ROOT"
mkdir -p "results/$MODE/$MODEL" "$HF_HOME" "$TRITON_CACHE_DIR"

export HF_HOME
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
unset MKL_INTERFACE_LAYER || true

echo "mode=$MODE model=$MODEL host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID}"
echo "model_root=$MODEL_ROOT dtype=${EFFECTIVE_MODEL_DTYPE:-model_default}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

selection=(--all)
if [[ "$MODE" == "test" ]]; then
    # One frozen question from each dataset; native VLMs therefore make
    # 2 questions x 8 cases x 30 patches = 480 patch calls per model.
    selection=(--qids TCGA-1 BCNB-31)
fi

model_args=(--hf-home "$HF_HOME")
if [[ "$MODEL" == "slidechat" ]]; then
    model_args+=(--checkpoint "$SLIDECHAT_CHECKPOINT")
fi
if [[ -n "$EFFECTIVE_MODEL_DTYPE" ]]; then
    model_args+=(--dtype "$EFFECTIVE_MODEL_DTYPE")
fi

"$PYTHON" scripts/run_experiment.py \
    --model "$MODEL" \
    "${selection[@]}" \
    --output-dir "results/$MODE/$MODEL" \
    "${model_args[@]}" \
    --fail-fast
