#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${OLLAMA_MODEL:-qwen2.5:0.5b}"
OLLAMA_API_URL="${OLLAMA_API_URL:-http://127.0.0.1:11434}"
FASTAPI_HOST="${FASTAPI_HOST:-0.0.0.0}"
FASTAPI_PORT="${FASTAPI_PORT:-8000}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-180}"

OLLAMA_PID=""

log() {
  local message="$1"
  printf '[entrypoint] %s\n' "$message"
}

cleanup() {
  if [[ -n "${OLLAMA_PID}" ]] && kill -0 "${OLLAMA_PID}" >/dev/null 2>&1; then
    log "Stopping Ollama process (${OLLAMA_PID})"
    kill "${OLLAMA_PID}"
    wait "${OLLAMA_PID}" || true
  fi
}

start_ollama() {
  log "Starting Ollama server"
  ollama serve >/tmp/ollama.log 2>&1 &
  OLLAMA_PID="$!"
}

wait_for_ollama() {
  log "Waiting for Ollama API at ${OLLAMA_API_URL}"
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    if curl -fsS "${OLLAMA_API_URL}/api/tags" >/dev/null 2>&1; then
      log "Ollama API is available"
      return 0
    fi

    if ! kill -0 "${OLLAMA_PID}" >/dev/null 2>&1; then
      log "Ollama process exited unexpectedly"
      cat /tmp/ollama.log || true
      exit 1
    fi

    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - start_ts > WAIT_TIMEOUT_SECONDS )); then
      log "Timeout while waiting for Ollama startup"
      cat /tmp/ollama.log || true
      exit 1
    fi

    sleep 1
  done
}

ensure_model() {
  log "Checking model ${MODEL_NAME}"
  if ollama list | awk 'NR>1 {print $1}' | grep -Fxq "${MODEL_NAME}"; then
    log "Model ${MODEL_NAME} already exists"
    return 0
  fi

  log "Pulling model ${MODEL_NAME}"
  ollama pull "${MODEL_NAME}"
}

start_fastapi() {
  log "Starting FastAPI on ${FASTAPI_HOST}:${FASTAPI_PORT}"
  exec uvicorn llm_service.app.main:app --host "${FASTAPI_HOST}" --port "${FASTAPI_PORT}"
}

main() {
  trap cleanup EXIT INT TERM
  start_ollama
  wait_for_ollama
  ensure_model
  start_fastapi
}

main "$@"
