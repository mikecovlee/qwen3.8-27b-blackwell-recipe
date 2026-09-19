#!/usr/bin/env bash
# Run on an AIR-GAPPED machine: install from a bundle/ built by export.sh.
#   scripts/offline/setup.sh [bundle_dir]     (default: ./bundle)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BUNDLE="${1:-bundle}"
MODEL_DIR_NAME="${MODEL_DIR:-Qwen3.8-27B-NVFP4}"

log() { printf '\033[1;34m[offline]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[offline] error:\033[0m %s\n' "$*" >&2; exit 1; }

VARIANT="${VARIANT:-$(cat "${BUNDLE}/VARIANT" 2>/dev/null || true)}"
VARIANT="${VARIANT:-fp8v}"
case "$VARIANT" in
  nvfp4) COMPOSE="inference/kv-nvfp4-text-image.yml" ;;
  fp8)   COMPOSE="inference/kv-fp8-text-only.yml" ;;
  fp8v)  COMPOSE="inference/kv-fp8-text-image.yml" ;;
  *) die "unknown VARIANT '$VARIANT' (use nvfp4 | fp8 | fp8v)" ;;
esac
if [[ "$VARIANT" == "fp8v" ]] && command -v nvidia-smi >/dev/null; then
  mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')"
  if [[ -n "$mem" ]]; then
    [[ "$mem" -ge 30000 ]] || die "VARIANT=fp8v needs a 32 GB+ GPU (detected: ${mem} MiB)"
  fi
fi

[[ -d "$BUNDLE" ]] || die "bundle not found: $BUNDLE (run export.sh on a networked machine first)"
command -v docker >/dev/null || die "docker not found"
docker compose version >/dev/null 2>&1 || die "docker compose v2 not found"

log "verifying checksums"
( cd "$BUNDLE" && sha256sum -c checksums.sha256 )

log "loading images"
docker load -i "$BUNDLE/images/sglang.tar"
docker load -i "$BUNDLE/images/new-api.tar"

if [[ ! -f .env ]]; then
  log "creating .env from .env.example"
  cp .env.example .env
  secret="$(openssl rand -hex 32 2>/dev/null || head -c32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  sed -i "s#^SESSION_SECRET=.*#SESSION_SECRET=${secret}#" .env
  sed -i "s#^MODELS_DIR=.*#MODELS_DIR=${HOME}/models#" .env
fi
if [[ "$VARIANT" == "fp8v" ]]; then
  expected_image="llm-infer:hicache-06e4f2ed"
else
  expected_image="lmsysorg/sglang@sha256:b91d664a8e4825afc16ab831c6035a6c88ac20ef8bd26da4fe2b9813a9f44376"
fi
if grep -q '^SGLANG_IMAGE=' .env; then
  sed -i "s#^SGLANG_IMAGE=.*#SGLANG_IMAGE=${expected_image}#" .env
else
  echo "SGLANG_IMAGE=${expected_image}" >> .env
fi
set -a; source .env; set +a
: "${MODELS_DIR:?set MODELS_DIR in .env}"

log "extracting model -> ${MODELS_DIR}/${MODEL_DIR_NAME}"
mkdir -p "$MODELS_DIR"
tar -C "$MODELS_DIR" -xf "$BUNDLE/model/${MODEL_DIR_NAME}.tar"

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
