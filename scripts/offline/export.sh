#!/usr/bin/env bash
# Run on a NETWORKED machine: build a self-contained bundle/ for air-gapped installs.
# Transfer the whole bundle/ directory to the offline host, then run
#   scripts/offline/setup.sh /path/to/bundle
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BUNDLE="${1:-bundle}"
SGLANG_DIGEST="lmsysorg/sglang@sha256:b91d664a8e4825afc16ab831c6035a6c88ac20ef8bd26da4fe2b9813a9f44376"
SGLANG_TAG="lmsysorg/sglang:dev-qwen38-27b-dflash2"
NEWAPI_IMAGE="${NEWAPI_IMAGE:-calciumion/new-api:v1.0.0-rc.36}"
MODEL_DIR_NAME="${MODEL_DIR:-Qwen3.8-27B-NVFP4}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

log() { printf '\033[1;34m[export]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[export] error:\033[0m %s\n' "$*" >&2; exit 1; }

[[ -d "${MODELS_DIR}/${MODEL_DIR_NAME}" ]] || die "model not found: ${MODELS_DIR}/${MODEL_DIR_NAME}"
mkdir -p "$BUNDLE/images" "$BUNDLE/model"

img="$SGLANG_DIGEST"
log "pulling SGLang image"
if ! docker pull "$img"; then
  img="$SGLANG_TAG"
  log "digest unavailable, using tag ${SGLANG_TAG}"
  docker pull "$img"
fi
log "pulling gateway image"
docker pull "$NEWAPI_IMAGE"

log "saving images (this is large, ~70 GB total with the model)"
docker save "$img" -o "$BUNDLE/images/sglang.tar"
docker save "$NEWAPI_IMAGE" -o "$BUNDLE/images/new-api.tar"

log "packing model ${MODEL_DIR_NAME}"
tar -C "$MODELS_DIR" -cf "$BUNDLE/model/${MODEL_DIR_NAME}.tar" "$MODEL_DIR_NAME"

log "writing checksums"
( cd "$BUNDLE" && sha256sum images/*.tar model/*.tar > checksums.sha256 )

cat > "$BUNDLE/README-offline.md" <<'EOF'
# Offline bundle

Built by `scripts/offline/export.sh`. Contents:

- `images/sglang.tar`, `images/new-api.tar` — Docker images (`docker load`)
- `model/<model-dir>.tar` — model weights
- `checksums.sha256` — verify with `sha256sum -c checksums.sha256`

Install on the air-gapped host:

1. Copy this whole `bundle/` directory and this repository to the host.
2. From the repository root run:

   ```bash
   scripts/offline/setup.sh /path/to/bundle
   ```
EOF

log "done -> $BUNDLE"
du -sh "$BUNDLE"/* 2>/dev/null || true
