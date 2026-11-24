#!/bin/bash
# Queue Qwen 3.8 27B on one GPU (slurm/serve_qwen.slurm):
#   bash slurm/submit_qwen.sh <h200|a100> [smoke|serve] [more sbatch options]
#   smoke: 75 min, start, test, stop.     serve: 2 h, keep serving.
# Both run in gpu-short (2-h cap; 2 GPUs and 2 running jobs per user). Options given after the mode win, e.g. a longer
# server in the gpu partition:  bash slurm/submit_qwen.sh h200 serve --partition=gpu --time=08:00:00
# Env knobs pass through to the job, for example:  HOLD_MIN=5 bash slurm/submit_qwen.sh a100 smoke
set -euo pipefail
GPU=${1:-}; MODE=${2:-smoke}
case $GPU in
  a100) RES=(--gres=gpu:a100:1 --constraint=a100@80g) ;;   # without the constraint it can land on a smaller A100
  h200) RES=(--gres=gpu:h200:1) ;;
  *) echo "usage: bash slurm/submit_qwen.sh <h200|a100> [smoke|serve] [more sbatch options]" >&2; exit 2 ;;
esac
case $MODE in
  smoke) PART=(--partition=gpu-short --time=01:15:00) ;;   # room for a cold start (up to 45 min) plus the tests
  serve) PART=(--partition=gpu-short --time=02:00:00) ;;   # gpu-short's cap
  *) echo "the mode is smoke or serve, not '$MODE'" >&2; exit 2 ;;
esac
shift $(( $# < 2 ? $# : 2 ))
cd "$(dirname "$0")/.."                                    # the repo root: the job's logs/ and scripts/ are relative
mkdir -p logs
sbatch "${PART[@]}" "${RES[@]}" --job-name="qwen_${MODE}_${GPU}" --export=ALL,MODE="$MODE" "$@" slurm/serve_qwen.slurm
