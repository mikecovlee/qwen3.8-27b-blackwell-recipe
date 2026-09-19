#!/usr/bin/env bash
# Consistency and hygiene checks. `--health` probes running services instead.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE=".env"
[[ -f "$ENV_FILE" ]] || ENV_FILE=".env.example"

if [[ "${1:-}" == "--health" ]]; then
  set -a; source "$ENV_FILE" 2>/dev/null || true; set +a
  sg="${SGLANG_PORT:-8080}"; gw="${GATEWAY_PORT:-8088}"; rc=0
  if curl -fsS --max-time 5 "http://localhost:${sg}/health" >/dev/null 2>&1; then
    echo "inference : OK   http://localhost:${sg}"
  else
    echo "inference : DOWN http://localhost:${sg}"; rc=1
  fi
  if curl -fsS --max-time 5 "http://localhost:${gw}/api/status" >/dev/null 2>&1; then
    echo "gateway   : OK   http://localhost:${gw}"
  else
    echo "gateway   : DOWN http://localhost:${gw}"; rc=1
  fi
  exit $rc
fi

fail=0

echo "== compose variants: shared-argument consistency =="
python3 - "$ENV_FILE" <<'PY' || fail=1
import difflib, json, subprocess, sys

env = sys.argv[1]
files = ["inference/kv-nvfp4-text-image.yml", "inference/kv-fp8-text-only.yml",
         "inference/kv-fp8-text-image.yml"]

def command(path):
    out = subprocess.run(
        ["docker", "compose", "--env-file", env, "-f", path, "config", "--format", "json"],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)["services"]["sglang"]["command"]

# Flags that exist in only one variant (drop flag + its single value).
DROP = {"--json-model-override-args", "--prefill-attention-backend", "--decode-attention-backend",
        "--attention-backend", "--mm-process-config",
        "--image-processor-backend", "--limit-mm-data-per-request",
        "--schedule-policy", "--chunked-prefill-size"}
# Boolean flags that exist in only one variant (no value to consume).
DROP_BOOL = {"--disable-prefill-cuda-graph"}
# Flags shared but intentionally different between variants (normalize the value).
# mrr / mamba pool / graph bs move TOGETHER per profile (three-knob rule, see README
# Tuning): fp8-text-image (mainline, v0.5.20 tree) runs 2/10/2 on 32 GB with the
# extra_buffer strategy; the legacy profiles (old pinned tree) run 4/16/4 + lazy.
NORM = {"--kv-cache-dtype", "--mem-fraction-static", "--max-running-requests",
        "--max-mamba-cache-size", "--cuda-graph-max-bs-decode",
        "--mamba-radix-cache-strategy"}

def norm(cmd):
    out, i = [], 0
    while i < len(cmd):
        tok = cmd[i]
        if tok in DROP_BOOL:
            i += 1
        elif tok in DROP:
            i += 2
        elif tok in NORM:
            out += [tok, "<variant>"]; i += 2
        else:
            out.append(tok); i += 1
    return out

cmds = [(f, norm(command(f))) for f in files]
ref_name, ref = cmds[0]
ok = True
for name, cmd in cmds[1:]:
    if cmd != ref:
        ok = False
        print(f"  MISMATCH: {name} vs {ref_name}:")
        for line in difflib.unified_diff(ref, cmd, ref_name, name, lineterm=""):
            print("   ", line)
if ok:
    print("  OK: shared arguments identical across all variants")
    sys.exit(0)
sys.exit(1)
PY

echo "== hygiene: secret scan =="
# Generic secret patterns. Extend with your own (ERE, '|'-joined) via HYGIENE_EXTRA.
PATTERNS='SESSION_SECRET=[0-9a-fA-F]{16,}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----'
[[ -n "${HYGIENE_EXTRA:-}" ]] && PATTERNS="${PATTERNS}|${HYGIENE_EXTRA}"
if grep -rInE "$PATTERNS" \
     --exclude-dir=.git --exclude-dir=bundle --exclude='.env' --exclude='check.sh' . ; then
  echo "  FAIL: secrets found (see above)"; fail=1
else
  echo "  OK: none"
fi

exit $fail
