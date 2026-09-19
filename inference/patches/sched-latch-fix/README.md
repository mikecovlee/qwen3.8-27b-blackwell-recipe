# sched-latch-fix — false-latch scheduler patches, one build per pinned image

## Layout

```
img-<first8-of-image-digest>/   one directory per serve-image digest
  ├─ sitecustomize.py           mount-based build (must keep this exact filename)   → img-b91d664a
  └─ sched_latch_fix.py         baked build (free name, loaded via .pth)            → img-d6e72886
     sched_lpm_waitfix.py       optional companion: bounded wait under LPM (see below)
     Dockerfile                  derived-image recipe (baked builds only)
```

`sched_lpm_waitfix.py` fixes LPM's known starvation gap (upstream sorts by prefix-hit
length with no aging — a cache-cold request can be overtaken forever while hot
continuations keep arriving). After each `calc_priority` pass it prepends the N
longest-waiting requests whose wait exceeds T (same `time.perf_counter` clock as
`time_stats.wait_queue_entry_time`). Enable per deployment via env:
`SGLANG_LPM_WAIT_BOOST_SECONDS=<T>` (0/unset = pure LPM), `SGLANG_LPM_WAIT_BOOST_MAX=<N>`
(1 = one boost per pass, matching the single-chunked-prefill admission budget).
Active only when the resolved policy is LPM; every internal error degrades to a
one-time loud warning then no-op. Retire when upstream adds aging/fairness to LPM.

| Serve image (compose `image:`) | Build | How it loads |
|---|---|---|
| `lmsysorg/sglang@sha256:b91d664a…` (dev tree `5f55db35`, 2026-08-22) | `img-b91d664a/sitecustomize.py` | bind-mount the dir as `/patches` + `PYTHONPATH=/patches` (all profiles here do this) |
| `lmsysorg/sglang@sha256:d6e72886…` (= `latest`, tree `0bcd822`, v0.5.19, 2026-09-04) | `img-d6e72886/` | derived image via its `Dockerfile`: module + `99-sched-latch-fix.pth` baked into site-packages |
| `lmsysorg/sglang@sha256:06e4f2ed…` (tree `94602c9`, v0.5.20, 2026-09-18) | `img-06e4f2ed/` | same bake, anchor re-verified at scheduler.py:3887 (double-count gate gained `candidate_beam_width`; shape unchanged). **`sched_lpm_waitfix` intentionally NOT baked**: v0.5.20 ships `--schedule-policy hrrn` (upstream aging) — observe that first |

Transitional copy note: the repo-root `sitecustomize.py` is a byte-identical copy of the
b91d664a build so that containers created before this restructure (they bind-mount the
parent dir itself) keep the patch across restarts until the next `docker compose up -d`
recreates them against `img-b91d664a/`. Safe to delete once no pre-restructure container
is in use; never edit it in place — the `img-*` dirs are canonical.

Each build pins its anchor in `scheduler.py`: b91d664a → line 3355, d6e72886 → line 3661
(both the `running_batch.batch_is_full = True` assignment behind the
`len(adder.can_run_list) >= get_num_allocatable_reqs(...)` check).

⚠ Why the two load differently: the `latest` base ships Ubuntu's
`/usr/lib/python3.12/sitecustomize.py` (apport hook), which satisfies Python's startup
`import sitecustomize` before `/patches` is searched — a PYTHONPATH-mounted patch would
never load, silently. So new-image builds must use the baked `.pth` route, where the
module also defers installation through a one-shot meta-path watcher (the `99-` file sorts
before the editable-install finder's `__editable__.sglang-*.pth`, so an eager
`import sglang` at .pth time fails) and self-verifies its anchor line at install time
(drift → loud `INSTALL FAILED` listing candidate lines). The legacy b91d664a build predates
both mechanisms, works via PYTHONPATH, and is frozen as-is. New builds follow the
`img-d6e72886` pattern.

## What the bug is

`scheduler.py` gates new prefill admits with
`len(adder.can_run_list) >= get_num_allocatable_reqs(running_bs)`. `can_run_list` always
contains the chunked-prefill continuation (`add_chunked_req` appends it unconditionally —
a memory-leak guard), while the quota side already discounts the rows that continuation
holds. Double counting latches `batch_is_full` spuriously until the next finish, so a
request arriving during a long chunked prefill waits out the whole prefill
(`max-running-requests = N` degrades to `N-1` under load).

The patch keeps the latch only when genuinely full: it snapshots free rows at pass start,
counts real new admits, and suppresses the latch set when `new_admits < rows_free`.
Verified against the false-latch concurrency logs in `local/archive/logs/` of the
maintainer's deployment (mrr-2 wall removed, no other behavior change).

## Adding a build for a new image digest `sha256:XXXXXXXX…`

1. `mkdir img-XXXXXXXX`, copy `img-d6e72886/sched_latch_fix.py` and `img-d6e72886/Dockerfile`
   into it; update the `FROM` digest and the log prefix.
2. Find the latch line:
   `docker run --rm --network=none --entrypoint bash <image> -c "grep -n 'running_batch.batch_is_full = True' /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py"`
   — pick the one inside `_get_new_batch_prefill_raw` whose preceding `if` compares
   `len(adder.can_run_list) >= self.get_num_allocatable_reqs(...)` (NOT the
   `get_num_allocatable_reqs() <= 0` pre-loop guard, NOT the `AddReqResult.NO_TOKEN` path).
3. Set `_ANCHOR_LINENO` accordingly. The self-check makes a wrong guess loud, not silent.
4. Smoke-test GPU-less (expect `armed`, then `anchor verified` + `installed`):
   `docker build -f img-XXXXXXXX/Dockerfile -t sglang-latchfix:XXXXXXXX .` (context = this
   directory), then
   `docker run --rm --network=none --entrypoint python3 sglang-latchfix:XXXXXXXX -c "import sglang.srt.managers.scheduler"`.
5. Point the profile at the new digest, then re-run the false-latch acceptance: 2 streams
   where A does a long chunked prefill (~48K-token prompt) and B arrives mid-prefill —
   B must start within one pass, not after A finishes.

If a future upstream release fixes the double counting outright, retire the patch instead
of chasing anchors (watch for `can_run_list` vs allocatable-quota semantics changing).
