# hicache-mamba-fix — HiCache hybrid-Mamba host-tier fix, one build per pinned image

## Layout

```
img-<first8-of-image-digest>/   one directory per serve-image digest
  ├─ patches/                   context diffs against the image's
  │                             /sgl-workspace/sglang checkout (git format,
  │                             one file per modified source file)
  └─ Dockerfile                  all-in-one derived image recipe: sched-latch
                                modules + these patches, in one build
```

The all-in-one image is built from `inference/patches/` — the context must include
both patch families:

```
cd inference/patches
docker build -f hicache-mamba-fix/img-06e4f2ed/Dockerfile -t llm-infer:hicache-06e4f2ed .
docker run --rm --network=none --entrypoint python3 llm-infer:hicache-06e4f2ed \
  -c "import sglang.srt.managers.scheduler"   # expects the sched-latch self-check
                                              # to log 'anchor verified' + 'installed'
```

## What the bug is

On hybrid GDN/Mamba models long prompts systematically missed the host (L2) KV
tier, so every eviction paid a full re-prefill:

1. **Chunked prefills never reached write-through backup.** The hit counter is
   advanced per scheduling operation, and chunked continuations are exempt from
   the threshold path, so any prompt longer than `--chunked-prefill-size` was
   permanently ineligible for backup (upstream #33714, PR #36647). `[P1]`
2. **The host-hit metric was a phantom.** `host_hit_length` counts radix hits,
   not materialized tokens: a request served entirely from *device* cache
   reported host hits with no host→device transfer, hiding the miss. `[P2]`
   *(carried on the v0.5.19 build; **retired on v0.5.20** — absorbed upstream as
   `host_loaded_length` + `materialized_host_hit_len()`.)*
3. **A mamba slot shortage crashed the scheduler** instead of degrading
   (`assert slot is not None` in `_alloc_mamba_slot`, upstream #34975 /
   PR #36770). `[P3]`
4. **The host mamba pool was sized by an unrelated ratio**, capping load-back
   anchors far below what branch reuse needs. The working criterion is
   `kv_pool_tokens / chunked_prefill_size ≤ device_mamba_slots + host_mamba_slots`
   (upstream #36935); `[C]` adds `SGLANG_HICACHE_MAMBA_SIZE_GB` to size the host
   pool directly. Default profile: `262144 / 6144 ≈ 43 ≤ 10 + 88`.

## Patches (v0.5.19 build, against tree 0bcd822 — historical; the v0.5.20 build drops P2, see below)

| Patch | File(s) | What it does | Upstream |
|---|---|---|---|
| P1 | `unified_radix_cache.py` | write-through backup of the newly inserted chain for `chunked=True` inserts (guarded by `enable_hicache` / `not is_write_back` / `not node.backuped`) | backport of #36647 (open) |
| P2 | `schedule_batch.py`, `schedule_policy.py` | add `Req.loaded_host_hit_length` (set only after a real load-back, reset on match/retract); use it at the 4 tier-split call sites | semantics of #26976 (open, approved) |
| P3 | `mamba_component.py`, `metrics_collector.py` | `_try_alloc_mamba_slot()` returns `None` after one eviction attempt; the 3 callers in `prepare_for_caching_req` return 0 (skip caching); 60 s-throttled warning + `sglang:radix_cache_aux_alloc_failed_total`; match/load-back allocation stays fatal | backport of #36770 (open) |
| C | `environ.py`, `hybrid_pool_assembler.py` | `SGLANG_HICACHE_MAMBA_SIZE_GB` (float GB, 0 = off) overrides the host mamba pool size in `build_hybrid_mamba_stack` and `build_hybrid_mamba_swa_stack` | no upstream counterpart yet — needs an issue/RFC (watch #38644) |

## Measured on d6e72886 (2026-09-17, single RTX PRO 4500 32 GB, Qwen3.8-27B-NVFP4, fp8 KV, cps 6144, host mamba 7.06 GB = 88 slots)

| Scenario | Before | After |
|---|---|---|
| cold prefill 36.6K | 8.85 s | 8.85 s (unchanged) |
| device-cache resend | 0.27 s | 0.27 s |
| **evicted resend (36,616 tokens)** | **8.76 s full recompute** | **0.26 s** — `load_back_tokens{pool="kv"} +36,608`, `{pool="mamba"} +2`, honest `host_hit +36,608` |
| cached branch continuation (~45.5K) | 25.1 s cold | 3.09 s (mamba-anchor-gated device reuse) |
| 2 × 68.5K concurrent | — | both OK in 42.7 s, OOM/assert/dropped = 0 |
| criterion | 262144/8192 = 32 > 10 ⇒ no branch reuse | 43 ≤ 10 + 88 ✓ |

T2b geometry (36,616 = 5 × 6144 + 5,896) also covers the chunk-boundary case that
upstream #39745 (draft) addresses; if a boundary miss is ever observed, add a
dedicated acceptance case before backporting #39745.

Acceptance: `inference/tools/acceptance/verify-hicache-thrash.py` (host-reload
latency + a host-load-back metric gate — a pass requires
`sglang:load_back_tokens_total{pool="kv"}` to actually grow).
Evidence: `evidence/hicache-mamba-fix-0917/`.

## Adding a build for a new image digest

The patches are context diffs against their pinned tree (0bcd822 for the removed
v0.5.19 build, 94602c9 for the current one) and will usually not apply unchanged
to a newer tree. Procedure:

1. Start a container from the new image; confirm `git status` in
   /sgl-workspace/sglang is clean.
2. Regenerate the canonical patch set from a deployment tree that carries the fix
   (`git diff -- <file>` per file, git format), or port each hunk by hand.
   `git apply --check` must pass in the new image — mismatches fail loudly.
3. Create `img-<first8-of-digest>/` with the new patches + Dockerfile (update the
   `FROM` digest and the log prefix) and add a row to the table above.
4. GPU-less smoke: build + `docker run --rm --entrypoint python3 <image> -c
   "import sglang.srt.managers.scheduler"`, then the marker self-checks (also
   enforced at build time by the Dockerfile).
5. Re-run the acceptance suites on an isolated test instance (same profiles,
   different container name and host port) before switching a profile's `image:`.

## Retirement

| Hunk | Retire when |
|---|---|
| P1 | #36647 (or #39745) lands in the pinned release line |
| P2 | **RETIRED 2026-09-19**: v0.5.20 (06e4f2ed) absorbed it upstream as `Req.host_loaded_length` + `materialized_host_hit_len()`; the img-06e4f2ed build carries no P2 files |
| P3 | #36770 lands (still open as of v0.5.20; assert re-confirmed at components/mamba.py:495 after upstream's mamba_component.py -> mamba.py rename) |
| C | upstream grows an equivalent explicit host-mamba sizing knob (watch #38644) |
| sched-latch modules | upstream fixes the false-latch double count (re-checked on v0.5.20 tree 94602c9: still unfixed, 2026-09-19). LPM-side fairness: waitfix retired 09-19, upstream HRRN accepted as mainline |

## v0.5.20 build (img-06e4f2ed, tree 94602c9)

Re-anchored onto the v0.5.20 line: 5 files instead of 7 (P2 dropped as
upstream-absorbed; schedule_batch/schedule_policy patches removed), P3 follows
the `components/mamba_component.py` -> `components/mamba.py` rename, P1 re-placed
after the new `rotation_tail_declined` early-return in `cache_unfinished_req`.
Exported patches under `img-06e4f2ed/patches/`; all-in-one Dockerfile verified
against a clean checkout (git apply --check) and baked as
`llm-infer:hicache-06e4f2ed` — the mainline build since 2026-09-19
(`kv-fp8-text-image.yml`; the observation profile was folded into it and deleted).

These are backports carried until the upstream PRs in the retirement table land;
the img-06e4f2ed all-in-one is the supported production build.
