#!/usr/bin/env bash
# One-click ONLINE install: pull images, download the model, start both services.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SGLANG_BASE_DIGEST="lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"
DERIVED_IMAGE="llm-infer:hicache-d6e72886"
PATCH_DIR="inference/patches"
SGLANG_DIGEST_LEGACY="lmsysorg/sglang@sha256:b91d664a8e4825afc16ab831c6035a6c88ac20ef8bd26da4fe2b9813a9f44376"
SGLANG_TAG_LEGACY="lmsysorg/sglang:dev-qwen38-27b-dflash2"
NEWAPI_IMAGE="calciumion/new-api:v1.0.0-rc.36"
MODEL_REPO="nvidia/Qwen3.8-27B-NVFP4"
MODEL_DIR_NAME="Qwen3.8-27B-NVFP4"

log() { printf '\033[1;34m[online]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[online] error:\033[0m %s\n' "$*" >&2; exit 1; }

VARIANT="${VARIANT:-fp8v}"
case "$VARIANT" in
  nvfp4) COMPOSE="inference/kv-nvfp4-text-image.yml" ;;
  fp8)   COMPOSE="inference/kv-fp8-text-only.yml" ;;
  fp8v)  COMPOSE="inference/kv-fp8-text-image.yml" ;;
  *) die "unknown VARIANT '$VARIANT' (use nvfp4 | fp8 | fp8v)" ;;
esac

log "preflight (variant: $VARIANT)"
command -v docker >/dev/null || die "docker not found"
docker compose version >/dev/null 2>&1 || die "docker compose v2 not found"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found (install the NVIDIA driver)"
nvidia-smi >/dev/null 2>&1 || die "no usable NVIDIA GPU"
docker info 2>/dev/null | grep -qi nvidia || log "warning: nvidia container runtime not detected"
if [[ "$VARIANT" == "fp8v" ]]; then
  mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')"
  [[ -n "$mem" && "$mem" -ge 30000 ]] || die "VARIANT=fp8v needs a 32 GB+ GPU (detected: ${mem:-?} MiB)"
fi

if [[ ! -f .env ]]; then
  log "creating .env from .env.example"
  cp .env.example .env
  secret="$(openssl rand -hex 32 2>/dev/null || head -c32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  sed -i "s#^SESSION_SECRET=.*#SESSION_SECRET=${secret}#" .env
  sed -i "s#^MODELS_DIR=.*#MODELS_DIR=${HOME}/models#" .env
fi
set -a; source .env; set +a
: "${MODELS_DIR:?set MODELS_DIR in .env}"

if [[ "$VARIANT" == "fp8v" ]]; then
  log "pulling SGLang base image (${SGLANG_BASE_DIGEST})"
  docker pull "$SGLANG_BASE_DIGEST" || die "cannot pull base image ${SGLANG_BASE_DIGEST}"
  log "building patched derived image (${DERIVED_IMAGE}; sched-latch + hicache-mamba)"
  docker build -f "${PATCH_DIR}/hicache-mamba-fix/img-d6e72886/Dockerfile" -t "$DERIVED_IMAGE" "$PATCH_DIR"
  SGLANG_IMAGE_FINAL="$DERIVED_IMAGE"
else
  log "pulling SGLang image (${SGLANG_DIGEST_LEGACY})"
  if ! docker pull "$SGLANG_DIGEST_LEGACY"; then
    log "digest unavailable, falling back to tag ${SGLANG_TAG_LEGACY}"
    docker pull "$SGLANG_TAG_LEGACY"
    SGLANG_IMAGE_FINAL="$SGLANG_TAG_LEGACY"
  else
    SGLANG_IMAGE_FINAL="$SGLANG_DIGEST_LEGACY"
  fi
fi
if grep -q '^SGLANG_IMAGE=' .env; then
  sed -i "s#^SGLANG_IMAGE=.*#SGLANG_IMAGE=${SGLANG_IMAGE_FINAL}#" .env
else
  echo "SGLANG_IMAGE=${SGLANG_IMAGE_FINAL}" >> .env
fi
log "pulling gateway image (${NEWAPI_IMAGE})"
docker pull "$NEWAPI_IMAGE"

if [[ ! -d "${MODELS_DIR}/${MODEL_DIR_NAME}" ]]; then
  log "downloading ${MODEL_REPO} -> ${MODELS_DIR}/${MODEL_DIR_NAME} (~22 GB)"
  mkdir -p "$MODELS_DIR"
  if command -v hf >/dev/null; then
    hf download "$MODEL_REPO" --local-dir "${MODELS_DIR}/${MODEL_DIR_NAME}"
  elif command -v huggingface-cli >/dev/null; then
    huggingface-cli download "$MODEL_REPO" --local-dir "${MODELS_DIR}/${MODEL_DIR_NAME}"
  else
    die "install huggingface_hub first:  pip install -U 'huggingface_hub[cli]'"
  fi
else
  log "model already present at ${MODELS_DIR}/${MODEL_DIR_NAME}"
fi

log "starting services"
docker compose --env-file .env -f "$COMPOSE" up -d
docker compose --env-file .env -f gateway/docker-compose.yml up -d

log "waiting for the inference server (model load takes ~1-2 min) ..."
for _ in $(seq 1 60); do
  if curl -fsS --max-time 3 "http://localhost:${SGLANG_PORT:-8080}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

bash scripts/check.sh --health || log "services not healthy yet; check: docker logs -f llm-infer"
log "done.  inference=http://localhost:${SGLANG_PORT:-8080}  gateway=http://localhost:${GATEWAY_PORT:-8088}"
log "next: open the gateway, create a user + token, then see gateway/opencode-config.md"
