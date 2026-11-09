# hicache-mamba-fix — HiCache hybrid-Mamba host-tier fix, one build per pinned image

## Layout

```
img-<first8-of-image-digest>/   one directory per serve-image digest
  ├─ patches/                   context diffs against the image's
  │                             /sgl-workspace/sglang checkout (git format,
  │                             one file per modified source file)
  └─ Dockerfile                  all-in-one derived image recipe: these
                                patches in one build (per base tree)
```

The all-in-one image is built from `inference/patches/`:

```
cd inference/patches
docker build -f hicache-mamba-fix/img-f832b5c3/Dockerfile -t llm-infer:hicache-f832b5c3 .
docker run --rm --network=none --entrypoint python3 llm-infer:hicache-f832b5c3 \
  -c "import sglang.srt.managers.scheduler"   # build-time self-check asserts
                                              # the patch markers at image build
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
   *(**Superseded on v0.5.21**: the reworked mamba accounting keeps the
   ratio-derived host pool sufficient — T3/T4 pass without the C env — so `[C]`
   is carried baked-but-dormant, not set by the profile.)*

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
| P1 | **RETIRED since v0.5.21**: v0.5.21 (b1259f3e) absorbed it upstream — a first-class chunked-prefill stash + `flush_pending_backups()` backup pipeline supersedes our write-through hunk; the img-b1259f3e build carries no `unified_radix_cache.py` patch and T4 passes without it |
| P2 | **RETIRED 2026-09-19**: v0.5.20 (06e4f2ed) absorbed it upstream as `Req.host_loaded_length` + `materialized_host_hit_len()`; the img-06e4f2ed build carries no P2 files |
| P3 | #36770 lands (still open as of v0.5.23; ported onto the new `_alloc_mamba_slot(req)` signature on img-f832b5c3, build self-check green 2026-11-09) |
| C | **REQUIRED on v0.5.22 and v0.5.23**: v0.5.22's write-through backs up ~101 mamba checkpoints for the T4 workload (10 on v0.5.21), which overflows the default 1.65 GB host pool (~20 slots); the LRU drops session states and a host hit becomes a full replay (T4 REPLAY 125 s). `SGLANG_HICACHE_MAMBA_SIZE_GB=8.5` (~106 slots) restores T4 PASS (0.5 s, load_back +137344), matching v0.5.21. **T4 re-check 2026-11-09 on v0.5.23: still REQUIRED** (without the env D replays 125 s, load_back +0; #42701/#44800 did not shrink the working set). Keep until upstream reworks the host mamba pool sizing; re-check on each bump |
| sched-latch modules | **RETIRED since v0.5.23**: upstream PR #44401 ("Count a request's held req_to_token row once at admission", merged 2026-10-15) fixed the double counting — `get_num_allocatable_reqs` adds `num_held_rows` back to the pool term; the gate shape changed so the monkeypatch anchor self-check would refuse to install anyway. The img-f832b5c3 build bakes NO sched-latch module. LPM-side fairness: waitfix retired 09-19, upstream HRRN accepted as mainline |

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

## v0.5.21 build (img-b1259f3e, tree e00930c)

Re-anchored onto the v0.5.21 line: 4 files instead of 5 (P1 dropped as
upstream-absorbed; the chunked-prefill stash + `flush_pending_backups()` pipeline
supersedes our write-through hunk, confirmed by T4). P3 re-anchored at
`components/mamba.py:509` (callers at 586/599/606); `[C]` carried but dormant (the
profile no longer sets `SGLANG_HICACHE_MAMBA_SIZE_GB`). v0.5.21 defaults to the
Rust unified-radix core (`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust`), which our
config uses — all gates pass under it. Exported patches under
`img-b1259f3e/patches/`; baked as `llm-infer:hicache-b1259f3e` — the mainline
build since v0.5.21 (`kv-fp8-text-image.yml`).

## v0.5.22 build (img-e3a4f060, tree baa9bef)

Re-anchored onto the v0.5.22 line (index digest
`sha256:e3a4f060…`, built 2026-10-19): still 4 files. P3 re-applied with line
offsets (upstream #36770 still open). `[C]` re-anchored because
`hybrid_pool_assembler.py` restructured the host-size split into an
`if get_memory().hicache_size > 0:` block with a QSA kv re-scale step — the
override now sits right after that block in both `build_hybrid_mamba_stack`
and `build_hybrid_mamba_swa_stack` (and the `envs` import moved up with the
reordered import block); still dormant unless
`SGLANG_HICACHE_MAMBA_SIZE_GB` is set. P1 stays retired (the chunked-prefill
stash + `flush_pending_backups()` pipeline is still first-class upstream; the
Rust TreeCore still has no `node_by_id()`). Exported patches under
`img-e3a4f060/patches/`; all-in-one baked as `llm-infer:hicache-e3a4f060`
(build self-check green 2026-10-20). Acceptance on the isolated twin: **T1 PASS**
(latch-log 0→3), **T3 PASS** (`--expect safe`, 2/2), **T4 PASS** with
`SGLANG_HICACHE_MAMBA_SIZE_GB=8.5` (D 0.5 s, load_back +137344). Without the env
T4 REPLAYs on plain v0.5.22 too: v0.5.22's write-through backs up ~101 mamba
checkpoints (vs 10 on v0.5.21), overflowing the default 1.65 GB host mamba pool
(~20 slots), so the LRU drops the session states and the host tier becomes
unmatchable. The knob is now active in `kv-fp8-text-image.yml`. Evidence and
ruled-out causes: `../../evidence/hicache-v0522-regression-1020/`. The profile
switched to e3a4f060 on 2026-10-20; the spec/200k-mtp profile stays on b1259f3e
pending the same validation.

## v0.5.23 build (img-f832b5c3, tree 297e54f)

Re-anchored onto the v0.5.23 line (index digest
`sha256:f832b5c3…`, built 2026-11-09): still 4 files, but **P3 was ported, not
re-anchored** — `_alloc_mamba_slot` gained a `req` parameter and a lazy
extra-buffer lock-release step in v0.5.23, so the hunk was rewritten to merge
the `_try_alloc_mamba_slot()` None-skip semantics into the new signature (the
three donate-path callers now pass `req` and return 0 on failure; the
external-linker LOAD path at `mamba.py:807` keeps the fatal
`_alloc_mamba_slot()`). `[C]` re-applied with offsets (+76/+83); `_split_hicache_size`
now prefers `host_capacity_bytes` but the override insertion points are
unchanged. **sched-latch-fix is NOT baked into this build**: upstream PR
**#44401** ("Count a request's held req_to_token row once at admission",
merged 2026-10-15, in v0.5.23) fixed the double counting —
`get_num_allocatable_reqs` now adds `num_held_rows` back to the pool term at
the main gate (`scheduler.py:4225-4232`) and the dLLM gate; the PD-prefill gate
that #44401 also fixed was later refactored out of `_get_new_batch_prefill_raw`
entirely. Exported patches under `img-f832b5c3/patches/`; all-in-one baked as
`llm-infer:hicache-f832b5c3` (build self-check green 2026-11-09). Acceptance on
the isolated twin (2026-11-09): **T1 PASS** (B@168.5 = A prefill end,
latch-log 0→0 — the false-latch no longer occurs with no module loaded),
**T3 PASS** (safe, 2/2), **T4 PASS** with `SGLANG_HICACHE_MAMBA_SIZE_GB=8.5`
(D 0.4 s, load_back +137344); **without the env T4 REPLAYs** (D 124.5 s,
load_back +0) — the C knob is STILL REQUIRED on v0.5.23 (v0.5.23's
#42701/#44800 write-ordering fixes did not change the checkpoint working
set). Evidence: `../../evidence/sglang-0523-upgrade-1109/`.
