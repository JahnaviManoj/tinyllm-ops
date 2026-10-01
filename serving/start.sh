#!/bin/bash
# Container entrypoint (tutorial 4.5/4.6): get the model, start llama-server, wait, run the gateway.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/model.gguf}"
LLAMA_PORT="${LLAMA_PORT:-8081}"
CTX="${LLAMA_CTX:-1024}"
THREADS="${LLAMA_THREADS:-$(nproc)}"

if [ ! -s "$MODEL_PATH" ]; then
  : "${MODEL_URL:?no model at $MODEL_PATH and MODEL_URL is unset}"
  echo "start.sh: downloading model from MODEL_URL -> $MODEL_PATH"
  mkdir -p "$(dirname "$MODEL_PATH")"
  curl -fsSL --retry 3 "$MODEL_URL" -o "$MODEL_PATH.part"
  if [ -n "${MODEL_SHA256:-}" ]; then
    echo "$MODEL_SHA256  $MODEL_PATH.part" | sha256sum -c - >/dev/null
  fi
  mv "$MODEL_PATH.part" "$MODEL_PATH"
fi
if [ -n "${MODEL_INFO_URL:-}" ]; then  # 4.6: artifacts/champion/model.json — log which version booted
  curl -fsSL "$MODEL_INFO_URL" || true; echo
fi

# --cache-ram 0 / --ctx-checkpoints 0: on this recurrent architecture the server's prompt caches
# otherwise grow without bound (decisions.md 2026-10-01).
llama-server -m "$MODEL_PATH" --host 127.0.0.1 --port "$LLAMA_PORT" -c "$CTX" -t "$THREADS" \
  -np 1 --cache-ram 0 --ctx-checkpoints 0 &
LLAMA_PID=$!
trap 'kill $LLAMA_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null; then break; fi
  if ! kill -0 $LLAMA_PID 2>/dev/null; then echo "start.sh: llama-server died" >&2; exit 1; fi
  sleep 1
done
curl -sf "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null || { echo "start.sh: llama-server not healthy" >&2; exit 1; }
echo "start.sh: llama-server healthy on :$LLAMA_PORT (llama.cpp ${LLAMA_CPP_COMMIT:-?}); starting gateway on :8080"

exec uvicorn serving.app:app --host 0.0.0.0 --port 8080
