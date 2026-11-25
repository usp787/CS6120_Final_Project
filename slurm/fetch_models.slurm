#!/bin/bash -l
#SBATCH --job-name=fetch_models
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CPU only. Downloads the project's small models (about 4 GB) to scratch with scripts/fetch_models.py, then links the
# Qwen 3.8 27B weights that Gemma4_kaggle_repo already downloaded and hash-checked (55.6 GB, so they are not downloaded
# again). BANKING77 comes with the repo (data/banking77). Safe to rerun: complete files are skipped. Nothing is uploaded.
# From the repo root on Explorer:
#   mkdir -p logs && sbatch slurm/fetch_models.slurm

ROOT=/scratch/$USER/cs6120
VENV=$ROOT/venvs/tools                                 # only huggingface_hub; the project's own env comes later
QWEN_SRC=/scratch/$USER/gemma4/models/Qwen3.8-27B
QWEN_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0      # the revision Gemma4's fetch verified (its .verified file)

module load explorer python/3.13.5                     # explorer: the proxy compute nodes use to reach the internet
set -euo pipefail
export HF_HOME=$ROOT/hf_cache HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_DISABLE_PROGRESS_BARS=1
echo "job=${SLURM_JOB_ID:-local} node=$(hostname) start=$(date -Is) root=$ROOT"

if ! "$VENV/bin/python" -c "import huggingface_hub" 2>/dev/null; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet "huggingface_hub==1.33.0"   # the version Explorer's Qwen fetch used
fi

if ! "$VENV/bin/python" scripts/fetch_models.py --root "$ROOT"; then
  echo "WARN: some downloads failed; retrying once without Xet"
  HF_HUB_DISABLE_XET=1 "$VENV/bin/python" scripts/fetch_models.py --root "$ROOT"
fi

mkdir -p "$ROOT/models"
if [ "$(cat "$QWEN_SRC/.verified" 2>/dev/null || true)" = "$QWEN_REV" ]; then
  ln -sfn "$QWEN_SRC" "$ROOT/models/Qwen3.8-27B"
  echo "OK   Qwen/Qwen3.8-27B@${QWEN_REV:0:12}: linked $ROOT/models/Qwen3.8-27B -> $QWEN_SRC"
else
  echo "WARN: $QWEN_SRC is missing or unverified; fetch Qwen3.8-27B before serving"
fi
du -shD "$ROOT"/models/* 2>/dev/null || true
echo "done $(date -Is)"
