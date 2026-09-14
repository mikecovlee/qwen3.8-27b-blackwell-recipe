# sched-latch-fix — false-latch scheduler patches, one per pinned image

## Which file do I need?

| Serve image (compose `image:`) | How the patch gets in |
|---|---|
| `lmsysorg/sglang@sha256:b91d664a…` (dev tree `5f55db35`, 2026-08-22) | bind mount + `PYTHONPATH=/patches` (root `sitecustomize.py`) — what all profiles here do today |
| `lmsysorg/sglang@sha256:d6e72886…` (= `latest` tag, tree `0bcd822`, v0.5.19, 2026-09-04) | **derived image** via [`Dockerfile.latest`](Dockerfile.latest): module + `99-sched-latch-fix.pth` baked into site-packages (`img-d6e72886/sched_latch_fix.py`) |

The root `sitecustomize.py` is the historically shipped build for the `b91d664a` image
(anchor: `scheduler.py:3355`). All profiles in this repo mount `PYTHONPATH=/patches`.

⚠ The `PYTHONPATH` method **cannot work** on the `latest` base: it ships Ubuntu's
`/usr/lib/python3.12/sitecustomize.py` (apport hook), which satisfies Python's startup
`import sitecustomize` before `/patches` is reached — a mounted patch module never loads,
silently. Bake it in instead. Conversely the baked `.pth` must not import sglang eagerly
(our `99-` sorts before the editable-install finder's `__editable__.sglang-*.pth`), which
is why `img-d6e72886/sched_latch_fix.py` defers installation through a one-shot meta-path
watcher and self-verifies its anchor line at install time (drift → loud `INSTALL FAILED`).

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

## Upgrading to a new image

Anchor drift is the only failure mode, and the `img-*` builds refuse to install silently:
they verify the anchor statement's line content at startup and log
`INSTALL FAILED: anchor drift ... candidate lines in this tree: [...]`.

To re-anchor for a new digest `sha256:XXXXXXXX…`:

1. `mkdir img-XXXXXXXX && cp` the newest `img-*/sched_latch_fix.py` into it; copy
   `Dockerfile.latest` to `Dockerfile.img-XXXXXXXX` and update its `FROM` digest and copy path.
2. Find the latch line:
   `docker run --rm --network=none --entrypoint bash <image> -c "grep -n 'running_batch.batch_is_full = True' /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py"`
   — pick the one inside `_get_new_batch_prefill_raw` whose preceding `if` compares
   `len(adder.can_run_list) >= self.get_num_allocatable_reqs(...)` (NOT the
   `get_num_allocatable_reqs() <= 0` pre-loop guard, NOT the `AddReqResult.NO_TOKEN` path).
3. Update `_ANCHOR_LINENO`, the header comment, and the `_log` prefix in the new file.
4. Smoke-test without a GPU or network (expect `armed`, then `anchor verified` + `installed`):
   `docker run --rm --network=none -v "$PWD/sched-latch-fix/img-XXXXXXXX/sched_latch_fix.py:/opt/sglang/lib/python3.12/site-packages/sched_latch_fix.py:ro" -e PYTHONPATH_UNUSED=1 --entrypoint python3 <image> -c "import sglang.srt.managers.scheduler" 2>&1 | grep sched-latch`
   (or build the derived image and run `python3 -c pass` + the import line inside it)
5. Point the profile at the new digest and `PYTHONPATH`, then re-run the false-latch
   acceptance: 2 streams where stream A does a long chunked prefill (e.g. a ~48K-token
   prompt) and B arrives mid-prefill — B must start within one pass, not after A finishes.

If a future upstream release fixes the double counting outright, retire the patch instead
of chasing anchors (watch for `can_run_list` vs allocatable-quota semantics changing).
