#!/bin/bash -l
#SBATCH --job-name=build_rag_env
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CPU only. Builds the Python env for the retrieval and classification side from requirements.txt into
# /scratch/$USER/cs6120/venvs/rag, checks it with scripts/check_rag_env.py (imports, BANKING77, and one small CPU run
# of each local model), then writes the exact versions to requirements-lock.txt. A rerun reuses the env and installs
# only what changed; REBUILD=1 starts over (for example after a scratch purge damages it).
# The Qwen server keeps its own vLLM env (slurm/serve_qwen.slurm). From the repo root:
#   mkdir -p logs && sbatch slurm/build_rag_env.slurm

ROOT=/scratch/$USER/cs6120
VENV=$ROOT/venvs/rag

module load explorer python/3.13.5                     # explorer: the proxy compute nodes use to reach PyPI
set -euo pipefail
export PIP_CACHE_DIR=$ROOT/cache/pip PIP_DISABLE_PIP_VERSION_CHECK=1    # pip's cache stays off the full /home
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
echo "job=${SLURM_JOB_ID:-local} node=$(hostname -s) start=$(date -Is) venv=$VENV base=$(python3 --version)"

if [ "${REBUILD:-0}" = 1 ] || [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --clear "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -r requirements.txt
"$VENV/bin/python" -m pip check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$VENV/bin/python" scripts/check_rag_env.py --root "$ROOT"
"$VENV/bin/python" -m pip freeze --exclude-editable > requirements-lock.txt
echo "wrote requirements-lock.txt ($(wc -l < requirements-lock.txt) packages); env size $(du -sh "$VENV" | cut -f1)"
echo "Jupyter on Open OnDemand, once:  $VENV/bin/python -m ipykernel install --user --name cs6120-rag"
echo "done $(date -Is)"
