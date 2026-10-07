#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -eo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit through coare/submit.sh; do not run on the frontend.' >&2; exit 2; }
# Slurm executes a spooled script; its BASH_SOURCE path is not the package path.
ROOT=${HSDPA_ROOT:?Submit through coare/submit.sh}
source "${HSDPA_CONFIG:?}"
[[ -z "$CONDA_MODULE" ]] || module load "$CONDA_MODULE"
command -v conda >/dev/null || { echo 'Conda not on PATH; check CONDA_MODULE.' >&2; exit 2; }
eval "$(conda shell.bash hook)"
ACTION=$1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export YOLO_CONFIG_DIR="$ROOT/outputs/ultralytics_settings"
export YOLO_AUTOINSTALL=false
export YOLO_OFFLINE
export CONDA_PKGS_DIRS="$(dirname -- "$ENV_PREFIX")/pkgs"
if [[ "$ACTION" == setup ]]; then
    [[ ! -e "$ENV_PREFIX" ]] || { echo 'Environment exists. Choose a new ENV_PREFIX to rebuild deliberately.' >&2; exit 2; }
    conda create --yes --prefix "$ENV_PREFIX" python=3.10 pip
    conda activate "$ENV_PREFIX"
    set -u
    python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url "$TORCH_INDEX_URL"
    python -m pip install -r "$ROOT/coare/requirements.txt"
    # The wheel supplies framework metadata; all model imports use our verified runtime.
    python -m pip install --no-deps "$ROOT/vendor/ultralytics-8.3.18-py3-none-any.whl"
    if [[ -d "$ROOT/runtime" ]]; then
        python "$ROOT/coare/build_runtime.py" --verify
    else
        python "$ROOT/coare/build_runtime.py"
    fi
    python -m pip freeze > "$ROOT/outputs/environment_${SLURM_JOB_ID}.txt"
    echo 'Environment ready. Next submit a check or smoke job.'
    exit 0
fi
conda activate "$ENV_PREFIX"
set -u
nvidia-smi
python "$ROOT/coare/build_runtime.py" --verify
if [[ "$ACTION" == check ]]; then
    srun python "$ROOT/coare/run_baseline.py" check
elif [[ "$ACTION" == resume ]]; then
    srun python "$ROOT/coare/run_baseline.py" resume --data "$DATA_YAML" --checkpoint "$2"
else
    srun python "$ROOT/coare/run_baseline.py" "$ACTION" --data "$DATA_YAML"
fi
