# SGLang v0.5.21 migration

Evidence for re-anchoring the patch build onto **SGLang v0.5.21** (base image
`lmsysorg/sglang:v0.5.21`, digest `sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`,
tree `e00930c`) producing `llm-infer:hicache-b1259f3e`, the mainline image since v0.5.21.
Patch build: [`inference/patches/`](../../inference/patches/).
Rebuilt on the v0.5.21 line (image id `ad595b31c52d`) with a diagnostic-only fix to the P3 warning
path (`_on_mamba_alloc_failed` now reads component sizes via the core-agnostic method, valid
under the Rust core); the acceptance numbers below are unchanged by that fix.

The v0.5.21 line is a large HiCache/Mamba rework plus a new default radix core (Rust).
Three of our four carried items were absorbed or obsoleted; two remain.

## Patch retirement status

| Patch | v0.5.20 (06e4f2ed) | v0.5.21 (b1259f3e) | Why |
| --- | --- | --- | --- |
| P1 chunked write-through | carried (#36647) | **RETIRED** | v0.5.21 ships a first-class chunked-prefill stash (`scheduler.stash_chunked_request` → `maybe_cache_unfinished_req(chunked=True)`) + `flush_pending_backups()`; the Rust tree core has no `node_by_id`, so the old hunk can't run anyway |
| P2 honest host-hit | retired in v0.5.20 | retired | absorbed upstream (`host_loaded_length` / `materialized_host_hit_len()`) |
| P3 mamba-alloc degrade | carried (#36770) | **CARRIED** | upstream still fatal-asserts at `unified_cache/components/mamba.py:509` (and `:214`); backport re-anchored |
| C mamba host-pool knob | carried (no upstream #) | **OBSOLETE** | v0.5.21 sizes the mamba host pool from the HiCache ratio (`hicache_auto_size` / ratio-derived). The patched code stays baked but dormant when `SGLANG_HICACHE_MAMBA_SIZE_GB` is unset |
| sched-latch | carried (anchor 3887) | **CARRIED** | upstream false-latch double-count still present; anchor re-derived to `scheduler.py:3976` |

## Re-anchoring

- **sched-latch**: anchor `running_batch.batch_is_full = True` inside
  `_get_new_batch_prefill_raw` moved `scheduler.py:3887 → 3976`. The `.pth`
  self-verification refuses to arm if the anchor drifts; it logged
  `anchor verified: .../scheduler.py:3976 / installed` at startup.
- **P3**: `components/mamba.py` `_alloc_mamba_slot` unchanged in shape (fatal assert
  at `:509`); the 3 `prepare_for_caching_req` call sites match. All four kept patches
  applied with `git apply` (offsets absorbed automatically); build self-check asserted
  `git status --short | wc -l == 4` and the P2/P1 markers absent.

## Twin acceptance (exact final profile, image `llm-infer:hicache-b1259f3e`)

Isolated twin instance (compose override, port 8099 — single-GPU exclusivity).
Profile = `inference/kv-fp8-text-image.yml` as shipped (`--hicache-ratio 2`, **no**
`SGLANG_HICACHE_MAMBA_SIZE_GB`). Boot (< 70 s to `/health`): `Tree Core:
RustUnifiedTreeCore`; kv host pool `524289 tokens, 17.18 GB`; mamba host pool
`1.65 GB` (page_first); `hicache_write_policy=write_through`.

| Suite | Result |
| --- | --- |
| `verify-scheduling-patches.py` (T1) | **PASS** — B decodes at 169.1 s while A's chunked prefill runs to 204.1 s (35 s window); latch suppressed 3× |
| `verify-mamba-stash.py --expect safe` (T3) | **PASS** — 2/2 safe trials; 0 crash lines, 0 restarts |
| `verify-hicache-thrash.py` (T4) | **PASS** — 244K cold 124 s → host reload; `load_back_tokens{pool="kv"}` +136,064; continuation 0.3 s; retract 0 / assert 0 |
| `tools/probe-bmm-fp8.py` (quant gate) | **PASS** — FP8 kernels bit-for-bit vs cutlass (`rel_err 0.0000`, 4 cases) |

Raw outputs: [`t1-scheduling.txt`](t1-scheduling.txt), [`t3-mamba-stash.txt`](t3-mamba-stash.txt),
[`t4-hicache-thrash.txt`](t4-hicache-thrash.txt), [`quant-gate.txt`](quant-gate.txt),
startup lines in [`boot.txt`](boot.txt).

## Sizing criterion — superseded

The v0.5.19/09-17 criterion (`262144 / cps ≤ device + host mamba slots`, which needed the
7 GB knob at cps 6144) no longer holds on v0.5.21: the reworked mamba accounting plus an
auto/ratio-sized host pool pass T4 with only ~10 device + ~20 host slots. Both experiments
passed on the new image — with the old `SGLANG_HICACHE_MAMBA_SIZE_GB=7.0` (mamba host
7.06 GB) and with it unset (mamba host 1.65 GB) — so the knob was dropped from the profile.
`cps 6144` remains load-bearing (8192 OOMs this profile on 32 GB).

## Quality / compatibility

- `tools/quality-spotcheck.py` on the new instance vs the 2026-09-11 baseline: parity.
  The only divergence (`math_mul`) reproduced as a one-off NVFP4 sampling garble — re-ran
  the item 3× on the live server and got the correct value each time (known NVFP4 caveat,
  not a kernel regression).
- Metrics: `/metrics` exposes every name the cockpit adapter consumes except
  `sglang:num_aborted_requests_total` (not emitted in this build but still defined at
  `observability/metrics_collector.py:1657`; the cockpit reads it as absent/0). Endpoints
  `/server_info`, `/health`, `/metrics` are unchanged; `/get_server_info` is now a
  deprecated alias of `/server_info` (cockpit uses `/server_info`).

## Reproduce

```sh
# derived image (context = inference/patches/)
cd inference/patches
docker build -f hicache-mamba-fix/img-b1259f3e/Dockerfile -t llm-infer:hicache-b1259f3e .
# GPU-less smoke (anchor/marker self-check runs at build time)
docker run --rm --network=none --entrypoint python3 llm-infer:hicache-b1259f3e \
  -c "import sglang.srt.managers.scheduler"
# twin instance (never against live traffic), then the gates
python3 inference/tools/acceptance/verify-scheduling-patches.py \
  --url http://localhost:8099/generate --container llm-infer-test
python3 inference/tools/acceptance/verify-mamba-stash.py \
  --url http://localhost:8099/generate --container llm-infer-test --expect safe
python3 inference/tools/acceptance/verify-hicache-thrash.py \
  --url http://localhost:8099/generate --container llm-infer-test
```
