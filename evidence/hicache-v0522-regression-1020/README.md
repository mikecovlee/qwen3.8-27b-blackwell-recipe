# v0.5.22 HiCache host-tier regression (2026-10-20)

T4 `verify-hicache-thrash.py` **PASSes on v0.5.21 (b1259f3e)** and **REPLAYs on
v0.5.22 (e3a4f060)** in the same environment, same profile, same test. The
upgrade to v0.5.22 is **blocked** by this until an upstream fix lands.

## Environment

- GPU: RTX PRO 4500 Blackwell 32 GB; driver/CUDA unchanged from production.
- Profile: `inference/kv-fp8-text-image.yml` (mrr 2, mamba pool 10,
  `extra_buffer`, chunked prefill 6144, FP8 KV 262144, HiCache ratio 2,
  `write_through` + `kernel` + `page_first`), isolated twin on :8099,
  `inference_sglang-cache` volume reused.
- Fresh instance per run (host pool empty); A/B runs sequential, GPU exclusive.
- T4 defaults: A 244400 chars (~220K tokens) cold, B/C 94600 chars each,
  D = A + a short question after eviction.

## Result

| build | D TTFT | load_back kv delta | D cached-token | verdict |
|---|---:|---:|---:|---|
| v0.5.21 `b1259f3e` (patched, production line) | **0.5 s** | **+137344** | 204928 | **PASS** |
| v0.5.22 `e3a4f060` (re-anchored patch set) | 124.9 s | +0 | 0 | REPLAY |
| v0.5.22 plain `lmsysorg/sglang:v0.5.22` (no local patches) | 125.0 s | +0 | 0 | REPLAY |
| v0.5.22 plain + `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python` | 124.9 s | +0 | 0 | REPLAY |

Logs: `t4-v0521-b1259f3e.log`, `t4-v0522-patched-e3a4f060.log`,
`t4-v0522-plain.log`, `t4-v0522-pycore.log`.

## What it rules out

- **Not the local patch set**: plain v0.5.22 fails identically without
  sched-latch/hicache patches.
- **Not the Rust TreeCore**: the Python core fails identically.
- **Not host capacity/pressure**: `hicache_host_used_tokens` 385533 of 524289,
  `hicache_dropped_tokens_total{reason=host_pressure}` 0,
  `{reason=write_through_unbacked_eviction}` 0.
- **Not backup**: `hicache_backup_tokens_total{pool="kv"}` reached 402493 in the
  failed run (A+B+C all backed up), same order as the passing v0.5.21 run
  (394173).
- **Not the host mamba pool size**: both builds allocate 1.65 GB
  (`Allocating 1.65 GB host memory for hierarchical Mamba cache`), and the mamba
  checkpoint counters are small on both (v0.5.21: backup 10 / load 2;
  v0.5.22: backup 101 / load 0).
- **Not server-arg defaults**: parsed server_args diff between the running
  v0.5.21 and plain v0.5.22 shows no hicache/radix/mamba differences (only
  unrelated new fields and `file_storage_path` default `sglang_storage` → None).

## Symptom

Backed up → evicted from device → retained in host (zero drops) → **no match on
return**: D's whole re-prefill logs `#cached-token: 0` on every chunk (36
chunks), i.e. the tree does not surface A's host-only nodes at all. This is the
same failure shape as upstream issue #33713 (MAMBA nodes pruned instead of
downgraded on device eviction), which was fixed before v0.5.21 — v0.5.22 appears
to have reintroduced or re-triggered it.

Suspect range in v0.5.22 (all merged between v0.5.21 and the release):
#43567 (lazy Mamba checkpoint invalidation), #43596 (Mamba host pools from
device buffer infos), #43868 (allocation-target FULL eviction), #43206 /
#43919 (unified-memory allocation/eviction changes), #43174 (split-node
write-back re-admission). Bisect inside the image checkout
(`/sgl-workspace/sglang`, full git tree at `baa9bef`) is possible.

## Next steps

1. Report upstream (issue) with this evidence; ask whether #43868's
   allocation-target FULL eviction or the host pool declaration refactor
   (#40608 stack) regressed hybrid-Mamba host matching.
2. When fixed (0.5.23 / main), rebuild `llm-infer:hicache-<digest>` via
   `inference/patches/hicache-mamba-fix/img-e3a4f060/Dockerfile` pattern
   (re-anchor if the tree moved) and re-run T1/T3/T4.
3. Do **not** point `kv-fp8-text-image.yml` at `e3a4f060` before T4 passes.
   The e3a4f060 build itself is green on T1 (latch-log 0→3) and T3
   (`--expect safe`, 2/2 trials) — only T4 blocks.

## Resolution (2026-10-20)

Root cause, found with a debug build that dumps `MatchResult` plus every node's
component state along the request path (`[DBG-MATCH]` / `[DBG-MAMBA]`):

- A's full-KV prefix was backed up and matched (`full_kv_hit_length=196608` on
  D's admission), but **no node carried a valid Mamba value on device or host**,
  so the Mamba validator rejected the whole path and `best_match_node` fell back
  to the root (`cached-token: 0`).
- v0.5.22's write-through backs up **101** mamba checkpoints for this workload
  (`hicache_backup_tokens_total{pool="mamba"}`; v0.5.21: 10). The default host
  mamba pool is only 1.65 GB (~20 slots at 0.08 GB/slot), so its LRU evicts the
  session checkpoints — the newest first-in-path states included. v0.5.21's
  10 backups fit, which is why the same test passed there.
- Not an upstream host-matching bug: the state simply is not there when the
  match runs.

Fix: activate the existing `[C]` patch knob — `SGLANG_HICACHE_MAMBA_SIZE_GB=8.5`
(~106 slots) in `inference/kv-fp8-text-image.yml`. Re-ran T4 on a fresh twin:
**PASS — D 0.5 s, load_back +137344, cached-token 204928, 0 retractions**, the
same numbers as v0.5.21. Full gate set on the final config
(`llm-infer:hicache-e3a4f060` + 8.5 GB): probe-bmm-fp8 PASS, T1 PASS
(latch-log 0→3), T3 PASS 2/2, T4 PASS (fresh instance, twice), cache-report OK,
b2 PASS, 600 s soak PASS (106 req, 0 errors). The profile was switched on
2026-10-20; the v0.5.21 image remains for rollback.
