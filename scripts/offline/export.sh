#!/usr/bin/env bash
# Run on a NETWORKED machine: build a self-contained bundle/ for air-gapped installs.
# Transfer the whole bundle/ directory to the offline host, then run
#   scripts/offline/setup.sh /path/to/bundle
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BUNDLE="${1:-bundle}"
VARIANT="${VARIANT:-fp8v}"
SGLANG_BASE_DIGEST="lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
DERIVED_IMAGE="llm-infer:hicache-f832b5c3"
SPEC_IMAGE="llm-infer:spec-f832b5c3"
NVFP4_IMAGE="llm-infer:nvfp4-f832b5c3"
PATCH_DIR="inference/patches"
NEWAPI_IMAGE="${NEWAPI_IMAGE:-calciumion/new-api:v1.0.0-rc.45}"
MODEL_DIR_NAME="${MODEL_DIR:-Qwen3.8-27B-NVFP4}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

log() { printf '\033[1;34m[export]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[export] error:\033[0m %s\n' "$*" >&2; exit 1; }

[[ -d "${MODELS_DIR}/${MODEL_DIR_NAME}" ]] || die "model not found: ${MODELS_DIR}/${MODEL_DIR_NAME}"
mkdir -p "$BUNDLE/images" "$BUNDLE/model"

img="$DERIVED_IMAGE"
log "pulling SGLang base image (v0.5.23 line)"
docker pull "$SGLANG_BASE_DIGEST" || die "cannot pull base image ${SGLANG_BASE_DIGEST}"
log "building patched derived image (${DERIVED_IMAGE}; hicache P3/C, sched-latch retired upstream)"
docker build -f "${PATCH_DIR}/hicache-mamba-fix/img-f832b5c3/Dockerfile" -t "$DERIVED_IMAGE" "$PATCH_DIR"
if [[ "$VARIANT" == "nvfp4" ]]; then
  img="$NVFP4_IMAGE"
  log "building NVFP4 host-transport image (${NVFP4_IMAGE}; derived + #46452 port)"
  docker build -f "${PATCH_DIR}/nvfp4-host-transport/img-f832b5c3/Dockerfile" -t "$NVFP4_IMAGE" "$PATCH_DIR"
elif [[ "$VARIANT" == "fp8vmtp" ]]; then
  img="$SPEC_IMAGE"
  log "building spec-decoding image (${SPEC_IMAGE}; derived + draft-share)"
  docker build -f "${PATCH_DIR}/spec-draft-share/img-f832b5c3/Dockerfile" -t "$SPEC_IMAGE" "$PATCH_DIR"
fi
echo "$VARIANT" > "$BUNDLE/VARIANT"
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
