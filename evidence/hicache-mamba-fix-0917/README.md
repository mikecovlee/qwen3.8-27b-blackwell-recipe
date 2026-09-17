# HiCache hybrid-Mamba fix & host-pool sizing (2026-09-17)

Evidence for the 2026-09-17 HiCache repair on the v0.5.19 line (base digest `d6e72886`):
chunked write-through backup, honest host-hit accounting, graceful mamba-alloc
degradation, the `SGLANG_HICACHE_MAMBA_SIZE_GB` knob, and the
`chunked-prefill-size 6144` + 7 GB host-pool sizing. Patch build:
[`inference/patches/hicache-mamba-fix/`](../../inference/patches/hicache-mamba-fix/README.md).

## Symptom

Production HiCache reuse in the host tier was effectively zero for this hybrid GDN
(Mamba) model: 26 h of traffic produced 17,344 KV backup tokens / 64 ops and a 190K
continuation still measured ~109 s (full re-prefill), while the hit counters *looked*
healthy. Three compounding upstream gaps:

1. Chunked prefills (any prompt longer than cps) were ineligible for write-through
   backup — the per-operation hit counter skips them (upstream #33714, root-caused in
   #36647), so their KV never entered the host pool and eviction meant full recompute.
2. Hybrid-Mamba KV can only be loaded back through a live mamba branch anchor, and a
   262144-token pool with the default cps 2048 needs ~128 anchors; upstream sizing
   provides 10 device + 20 host slots, the criterion fails (128 > 30), anchors get
   evicted and eviction leaves no load-back path (upstream #36935).
3. The device/host/storage tier split attributed device-resident tokens to the host
   tier (phantom `host_hit`), hiding both failures (upstream #26976 semantics).

## Patch set

| Patch | File(s) | What it does | Upstream |
| --- | --- | --- | --- |
| P1 | `mem_cache/unified_radix_cache.py` | Chunked inserts write through to the host tier immediately (ancestors already covered by `_build_backup_kv_action`) | #36647 |
| P2 | `managers/schedule_policy.py`, `managers/schedule_batch.py` | `Req.loaded_host_hit_length` = tokens actually restored; tier split no longer credits device hits to host | #26976 |
| P3 | `mem_cache/unified_cache/components/mamba_component.py`, `observability/metrics_collector.py` | Mamba-slot alloc failure degrades (skip this chunk's caching) instead of asserting; census via `sglang:radix_cache_aux_alloc_failed_total` | #36770 |
| C | `environ.py`, `mem_cache/hybrid_cache/hybrid_pool_assembler.py` | `SGLANG_HICACHE_MAMBA_SIZE_GB` explicitly sizes the host mamba pool | — (sizing knob) |

## Measured (single 32 GB Blackwell sm120, twin instance, shipped profile)

| Scenario | Before | After |
| --- | --- | --- |
| 68K cold prefill | 8.85 s | 8.85 s (unchanged) |
| device-resident re-send | 0.27 s | 0.27 s (unchanged) |
| **evicted re-send (36.6K)** | **8.76 s full recompute, host_hit 0** | **0.26 s — ~34×** |
| branch re-send (45.5K, L2 fork) | 25.1 s | 3.09 s |
| 2 × 68.5K concurrent prefills | — | 42.7 s each, 0 errors |

The evicted re-send is the case enforced by
`inference/tools/acceptance/verify-hicache-thrash.py`; it now requires a real host
load-back (`sglang:load_back_tokens_total{pool="kv"}` must increase), not just a fast
answer. Counters after the fix (raw dump: [`tier-metrics-after.txt`](tier-metrics-after.txt)):

| Counter | Value |
| --- | --- |
| `sglang:hicache_backup_tokens_total{pool="kv"}` | 667,426 |
| `sglang:load_back_tokens_total{pool="kv"}` | 36,608 — first real host→device KV load on this deployment |
| `sglang:load_back_tokens_total{pool="mamba"}` | 4 |
| `sglang:load_back_duration_seconds_count` | 2 |
| `sglang:radix_cache_aux_alloc_failed_total` | absent (never triggered) |
| `sglang:hicache_dropped_tokens_total` | 0 / 0 |
| `sglang:hicache_host_used_tokens` | 524,002 |

## Sizing criterion

Every `--chunked-prefill-size` tokens of prefix consumes one mamba anchor, so the pool
needs `262144/cps` of them. With the upstream default cps 2048 that is ~128, far beyond
the default 10 device + 20 host slots. Shipped sizing (`cps 6144`,
`SGLANG_HICACHE_MAMBA_SIZE_GB=7.0`):

`262144 / 6144 ≈ 43  ≤  10 (device, --max-mamba-cache-size) + 88 (host, 7.06 GB) = 98`

93 anchors were generated across the runs and all fit (93 ≤ 98) — no anchor eviction.
The KV host pool (17.18 GB, ratio 2) and the 262144-token device pool are unchanged;
the extra 7.06 GB is host RAM only. cps 8192 was tried first and OOM-killed the
scheduler under load; 6144 is the largest chunk that keeps two chunks inside the
default `--max-prefill-tokens 16384`.

## Notes

- Zero retractions/assertions throughout. The only log lines matching `error` are
  client-side HTTP 400s from a bad-vocab stress request (270K tokens > 262144 context,
  2026-09-17 15:43), not scheduler faults.
- 24 h watch items: OOM lines (cps 6144 is the first deviation above the upstream
  default), `radix_cache_aux_alloc_failed_total`, `load_back_tokens{pool="kv"}` growth,
  and TBT/queue under cps-induced prefill overlap.

## Twin acceptance (2026-09-17, image `llm-infer:hicache-d6e72886`)

All three suites pass on an isolated twin instance (compose override, port 8099 —
single-GPU exclusivity: never alongside production):

| Suite | Result |
| --- | --- |
| `verify-scheduling-patches.py` | PASS — B decodes at 160.0 s while A's chunked prefill runs until 193.8 s (34 s discrimination window, false latch suppressed 3×); LPM boost fires 2× (C TTFT 24.5 s) |
| `verify-mamba-stash.py --expect safe` | PASS — 2/2 safe trials (`used 8 / evictable 1 / available 1` on pool 10); 0 crash lines, 0 restarts |
| `verify-hicache-thrash.py` | PASS — A continuation TTFT 0.4 s; `load_back_tokens{pool="kv"}` +137,344 (real host load); retract 0 / assert 0 |

The thrash suite first reported a retraction because the startup `server_args` dump
contains `retraction_policy`; the counter now excludes that line (real retractions are
scheduler log lines). This makes the gate strictly stronger: a fast answer without a
host load-back delta now FAILs.

## Reproduce

```sh
# derived image (context = inference/patches/)
docker build -f hicache-mamba-fix/img-d6e72886/Dockerfile -t llm-infer:hicache-d6e72886 .
# GPU-less smoke (anchor/marker self-check runs at build time)
docker run --rm --network=none --entrypoint python3 llm-infer:hicache-d6e72886 \
  -c "import sglang.srt.managers.scheduler"
# twin instance (never against live traffic), then the acceptance gate
python3 inference/tools/acceptance/verify-hicache-thrash.py \
  --url http://localhost:8099/generate --container llm-infer-test
```
