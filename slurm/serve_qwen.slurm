#!/bin/bash -l
#SBATCH --job-name=qwen_server
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Serve Qwen 3.8 27B (BF16) on one 80 GB+ GPU with vLLM's OpenAI-compatible API. Thinking is off by default (a request
# can still turn it on through chat_template_kwargs), and the default sampling is Qwen's for non-thinking mode.
#   MODE=smoke  start, run scripts/smoke_qwen.py against it, stop. HOLD_MIN=n keeps the server up n more minutes.
#   MODE=serve  start, run a one-request check, then serve until the job ends.
# Each job writes its address and a fresh API key to /scratch/$USER/cs6120/run/qwen_endpoint_<job>.env (readable only
# by you); serve mode also points qwen_endpoint.env there. The file is removed when the job ends.
# Queue it with slurm/submit_qwen.sh, which sets the partition, time and GPU:
#   bash slurm/submit_qwen.sh a100 smoke     bash slurm/submit_qwen.sh h200 smoke     bash slurm/submit_qwen.sh a100 serve
# Env knobs: HOLD_MIN (0), MAX_MODEL_LEN (16384), VLLM_ENV (Gemma4_kaggle_repo's vllm_host: vLLM 0.19.1)

MODE=${MODE:-smoke}; HOLD_MIN=${HOLD_MIN:-0}
ROOT=/scratch/$USER/cs6120
MODEL=$ROOT/models/Qwen3.8-27B                         # a link to the checked copy (slurm/fetch_models.slurm)
QWEN_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
SERVED=qwen3.8-27b
VLLM_ENV=${VLLM_ENV:-/scratch/$USER/gemma4/venvs/vllm_host}
PY=$VLLM_ENV/bin/python
JOB=${SLURM_JOB_ID:-local}
RUN=$ROOT/run
VLOG=$RUN/vllm_$JOB.log
EP=$RUN/qwen_endpoint_$JOB.env

set -euo pipefail
case $MODE in smoke|serve) ;; *) echo "FAIL: MODE=$MODE; use smoke or serve"; exit 1 ;; esac
mkdir -p "$RUN"
export HF_HOME=$ROOT/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=$ROOT/cache/vllm TRITON_CACHE_DIR=$ROOT/cache/triton      # compile caches stay off the full /home
export XDG_CACHE_HOME=$ROOT/cache/xdg FLASHINFER_WORKSPACE_BASE=$ROOT/cache/flashinfer
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}   # unset, startup can stall
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export no_proxy="127.0.0.1,localhost,$(hostname -s)${no_proxy:+,$no_proxy}"; export NO_PROXY=$no_proxy

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) || true
[ -n "$GPU_NAME" ] || { echo "FAIL: no GPU in this job; queue it with slurm/submit_qwen.sh"; exit 1; }
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "job=$JOB mode=$MODE node=$(hostname -s) gpu=$GPU_NAME ($GPU_MIB MiB) driver=$DRIVER vllm_env=$VLLM_ENV start=$(date -Is)"
[ "${GPU_MIB:-0}" -ge 79000 ] || { echo "FAIL: BF16 Qwen 3.8 27B needs an 80 GB+ GPU; this one has $GPU_MIB MiB"; exit 1; }
[ "$(cat "$MODEL/.verified" 2>/dev/null || true)" = "$QWEN_REV" ] \
  || { echo "FAIL: $MODEL is missing or not the checked download; run slurm/fetch_models.slurm"; exit 1; }

PORT=$("$PY" -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
VLLM_API_KEY=$("$PY" -c 'import secrets; print(secrets.token_urlsafe(24))')
[ -n "$VLLM_API_KEY" ] || { echo "FAIL: no API key; vLLM would start without one"; exit 1; }
export VLLM_API_KEY                                    # by env, so `ps` doesn't show it
# Listens on all interfaces so an SSH tunnel through the login node can reach it; the API key keeps others out.
# --language-model-only: no vision tower. Prefix caching stays off: vLLM 0.19.1 calls it experimental for this
# hybrid model. presence_penalty can't be a server default in 0.19.1; clients that want Qwen's 1.5 send it.
ARGS=(--model "$MODEL" --served-model-name "$SERVED" --language-model-only
  --host 0.0.0.0 --port "$PORT" --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-16384}" --gpu-memory-utilization 0.90
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'
  --override-generation-config '{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}'
  --no-enable-prefix-caching)
"$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" > "$VLOG" 2>&1 &
VLLM_PID=$!
cleanup() { rm -f "$EP"; kill "$VLLM_PID" 2>/dev/null || true; sleep 5; kill -9 "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 143' TERM INT                               # at the time limit, still run cleanup

T0=$(date +%s)
until curl -sf --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "FAIL: vLLM exited during startup; the end of $VLOG:"; tail -n 40 "$VLOG"; exit 1; }
  [ $(( $(date +%s) - T0 )) -lt 1800 ] || { echo "FAIL: vLLM not healthy after 30 min; the end of $VLOG:"; tail -n 40 "$VLOG"; exit 1; }
  sleep 5
done
STARTUP=$(( $(date +%s) - T0 ))
KV=$(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$VLOG" | grep -oE '[0-9,]+' | tr -d , || true)
( umask 077; printf 'QWEN_BASE_URL=http://%s:%s/v1\nQWEN_API_KEY=%s\nQWEN_MODEL=%s\nQWEN_JOB=%s\n' \
    "$(hostname -s)" "$PORT" "$VLLM_API_KEY" "$SERVED" "$JOB" > "$EP" )
echo "vLLM ready after $STARTUP s at http://$(hostname -s):$PORT/v1; KV cache ${KV:-?} tokens; endpoint file $EP"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

META=(--meta gpu="$GPU_NAME" --meta driver="$DRIVER" --meta node="$(hostname -s)" --meta job="$JOB"
  --meta startup_s="$STARTUP" --meta kv_cache_tokens="${KV:-}" --meta vllm_env="$VLLM_ENV")
if [ "$MODE" = smoke ]; then
  RC=0
  "$PY" scripts/smoke_qwen.py --endpoint "$EP" "${META[@]}" --out "$RUN/smoke_$JOB.json" || RC=$?
  if [ "$HOLD_MIN" -gt 0 ]; then
    echo "holding the server for $HOLD_MIN min"; sleep $(( HOLD_MIN * 60 ))
  fi
  echo "done $(date -Is) smoke_exit=$RC"
  exit "$RC"
fi
"$PY" scripts/smoke_qwen.py --endpoint "$EP" "${META[@]}" --quick --out "$RUN/serve_check_$JOB.json"
ln -sfn "$(basename "$EP")" "$RUN/qwen_endpoint.env"
echo "serving until the job ends. From your laptop:  ssh -N -L 8000:$(hostname -s):$PORT explorer"
echo "  then use http://localhost:8000/v1 with QWEN_API_KEY from $EP (on Explorer: source $RUN/qwen_endpoint.env)"
wait "$VLLM_PID"
