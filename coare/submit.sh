#!/bin/bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export HSDPA_ROOT="$ROOT"
ACTION=${1:-}
case "$ACTION" in setup|check|smoke|train|resume) ;; *) echo 'Usage: bash coare/submit.sh setup|check|smoke|train|resume [checkpoint.pt]' >&2; exit 2;; esac
export HSDPA_CONFIG=${HSDPA_CONFIG:-"$ROOT/coare/coare.env"}
[[ -f "$HSDPA_CONFIG" ]] || { echo 'Copy coare/coare.env.example to coare/coare.env and fill the assigned values.' >&2; exit 2; }
source "$HSDPA_CONFIG"
validate() { [[ -n "${!1:-}" && "${!1}" != *REPLACE* ]] || { echo "Configure $1 first." >&2; exit 2; }; }
validate ENV_PREFIX
[[ "$CONDA_MODULE" != *REPLACE* ]] || { echo 'Configure CONDA_MODULE first.' >&2; exit 2; }
if [[ "$ACTION" == setup ]]; then
    for key in SETUP_PARTITION SETUP_QOS; do validate "$key"; done
    options=(--partition="$SETUP_PARTITION" --qos="$SETUP_QOS" --time="$SETUP_TIME" --cpus-per-task="$SETUP_CPUS" --mem="$SETUP_MEMORY")
else
    for key in GPU_PARTITION GPU_QOS GPU_GRES; do validate "$key"; done
    [[ "$ACTION" == check ]] || validate DATA_YAML
    duration=$SMOKE_TIME
    [[ "$ACTION" != train && "$ACTION" != resume ]] || duration=$TRAIN_TIME
    options=(--partition="$GPU_PARTITION" --qos="$GPU_QOS" --gres="$GPU_GRES" --time="$duration" --cpus-per-task="$GPU_CPUS" --mem="$GPU_MEMORY")
fi
[[ -z "$SLURM_ACCOUNT" ]] || options+=(--account="$SLURM_ACCOUNT")
checkpoint=${2:-}
if [[ "$ACTION" == resume ]]; then
    [[ -f "$checkpoint" ]] || { echo 'Supply the existing last.pt checkpoint path.' >&2; exit 2; }
    checkpoint=$(realpath -- "$checkpoint")
fi
mkdir -p "$ROOT/logs" "$ROOT/outputs"
exec sbatch "${options[@]}" --export=ALL --chdir="$ROOT" --job-name="pestnet_$ACTION" --output="$ROOT/logs/${ACTION}_%j.out" --error="$ROOT/logs/${ACTION}_%j.err" "$ROOT/coare/job.sh" "$ACTION" "$checkpoint"
