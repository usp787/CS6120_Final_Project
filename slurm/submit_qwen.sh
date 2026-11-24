#!/bin/bash
# Queue Qwen 3.8 27B on one GPU (slurm/serve_qwen.slurm):
#   bash slurm/submit_qwen.sh <a100|h200> [smoke|serve] [more sbatch options]
#   smoke: partition gpu-short, 45 min: start, test, stop.     serve: partition gpu, 8 h: keep serving.
# Env knobs pass through to the job, for example:  HOLD_MIN=5 bash slurm/submit_qwen.sh a100 smoke
set -euo pipefail
GPU=${1:-}; MODE=${2:-smoke}
case $GPU in
  a100) RES=(--gres=gpu:a100:1 --constraint=a100@80g) ;;   # without the constraint it can land on a smaller A100
  h200) RES=(--gres=gpu:h200:1) ;;
  *) echo "usage: bash slurm/submit_qwen.sh <a100|h200> [smoke|serve] [more sbatch options]" >&2; exit 2 ;;
esac
case $MODE in
  smoke) PART=(--partition=gpu-short --time=00:45:00) ;;
  serve) PART=(--partition=gpu --time=08:00:00) ;;
  *) echo "the mode is smoke or serve, not '$MODE'" >&2; exit 2 ;;
esac
shift $(( $# < 2 ? $# : 2 ))
cd "$(dirname "$0")/.."                                    # the repo root: the job's logs/ and scripts/ are relative
mkdir -p logs
sbatch "${PART[@]}" "${RES[@]}" --job-name="qwen_${MODE}_${GPU}" --export=ALL,MODE="$MODE" "$@" slurm/serve_qwen.slurm
