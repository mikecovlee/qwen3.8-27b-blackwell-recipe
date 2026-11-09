# nvfp4-host-transport — NVFP4 payload + scale transport over HiCache L2

Port of upstream PR **#46452** ("[Fix] hicache: preserve NVFP4 payload and
scales across host transfers", open on `main` as of 2026-11-09) onto the
v0.5.23 line, for `inference/kv-nvfp4-text-image.yml`.

## Why this exists

NVFP4 KV stores packed fp4 payload (2 elements/byte, `store_dtype=uint8`)
plus per-block scales in separate tensors (`k_scale_buffer`/`v_scale_buffer`,
`KVScaleLayout.TOKEN`). Two upstream defects made HiCache + NVFP4 unusable:

- **v0.5.21 and earlier**: SILENT CORRUPTION — the MHA host backup moved only
  `k_buffer`/`v_buffer`, never the scale buffers, so write-back/load-back lost
  scales and attention dequantized with stale ones (sgl-project/sglang#43239,
  #36121). The generic host pool also sized rows with a full-width formula
  (512 B/token vs the real 256 B + scales).
- **v0.5.23**: fails fast at startup (`pool_host/mha.py:166` —
  `MHA host transfers do not support KV scale layout 'token'`). Safer than
  v0.5.21's silent loss, but it blocks the profile outright.

Upstream fix PRs #36121 / #43240 / #46452 are all open; this patch set is the
#46452 transport fix backported so the profile can run the v0.5.23 line.

## What it changes (4 files)

| File | Change |
|---|---|
| `mem_cache/device_pool_info.py` | NVFP4 buffer descriptor carrying scale-buffer geometry alongside the packed payload |
| `mem_cache/memory_pool.py` | Emit the NVFP4 descriptor from the device pool (`quant_method.name in ("nvfp4", "fp4_mx_block16")`; NOT the store dtype — `is_float4_e2m1fn_x2` is False at runtime, a detection trap noted upstream) |
| `mem_cache/pool_host/mha.py` | Route NVFP4 pools to `MHATokenToKVPoolFP4Host` in `get_mha_host_pool_cls` |
| `mem_cache/pool_host/mha_fp4.py` | New host pool: host payload rows sized in device bytes (`head_dim // 2`, uint8) so JIT element sizes are square; block scales ride page-granular side buffers on the staged kernel path (`page_first` only); non-page-multiple transfers zero touched host scale pages; L3 flat-storage raises `NotImplementedError` |

## Build

```sh
# context = inference/patches/; build llm-infer:hicache-f832b5c3 first
docker build -f nvfp4-host-transport/img-f832b5c3/Dockerfile \
  -t llm-infer:nvfp4-f832b5c3 .
```

Self-check: exactly 8 modified/new files on the chain (4 hicache + 4 here),
`MHATokenToKVPoolFP4Host` present in both files, AST parse green.

## Retirement

Retire when upstream merges #46452 (or #43240) — drop this patch set on the
next base bump and re-run the acceptance battery below.

## Acceptance (2026-11-09, isolated twin, 262144 pool)

- T1 latch PASS (B@168.5 = A prefill end)
- T3 mamba stash PASS (safe 2/2, zero restarts)
- T4 REPLAY without `SGLANG_HICACHE_MAMBA_SIZE_GB` / PASS (D 0.3 s, load_back
  +135296) with 8.5 — env added to the profile, same C-knob requirement as
  the other profiles on the v0.5.22+ write path.

Evidence: `../../evidence/sglang-0523-upgrade-1109/`. Note upstream's own
byte-exact D2H→wipe→H2D round-trip battery (in #46452/#43239) is the deeper
correctness proof; T4 here proves the load-back path exercises end-to-end.
