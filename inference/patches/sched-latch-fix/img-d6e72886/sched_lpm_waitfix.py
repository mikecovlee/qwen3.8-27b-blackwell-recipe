# sched-lpm-waitfix — bounded-wait companion for --schedule-policy lpm
# Build for lmsysorg/sglang:latest pinned at index digest
#   sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9
#   (= source tree 0bcd822, v0.5.19 line, built 2026-09-04)
#
# Why: upstream LPM sorts the waiting queue by prefix-hit length with no aging —
# a cache-cold request can be overtaken forever while cache-hot continuations keep
# arriving (unbounded starvation; only guard is upstream's queue>128 FCFS fallback).
# Fix: after every calc_priority pass, take the N longest-waiting requests whose
# wait exceeds T (clock: time.perf_counter, same source as
# time_stats.wait_queue_entry_time) and stably prepend them. Hot requests drain
# fast, so a boosted request wins the next slot release and, while it is starved,
# stays pinned (it is always the oldest overtime entry). Result: every already-
# timed-out request is admitted within one slot release; cold-wave stampedes are
# capped at N per pass (N=1 == the system's per-pass prefill digestion: single
# chunked-prefill pipeline + max_prefill_tokens budget).
#
# Knobs (env, read once at install):
#   SGLANG_LPM_WAIT_BOOST_SECONDS=<T>   0/unset = feature off (pure LPM)
#   SGLANG_LPM_WAIT_BOOST_MAX=<N>       default 1
# Active only when SchedulePolicy resolves to CacheAwarePolicy.LPM; no-op under
# fcfs (already oldest-first) and under upstream's >128 FCFS fallback (harmless).
#
# Known edges (accepted):
#   - a boost may overtake LPM's in-batch-prefix dedup sinking -> one cache-build
#     surge; rare, cost bounded.
#   - ordering only; adds no capacity.
#   - LPM is an input-side SJF proxy; long-output tasks are underestimated.
# Retire when upstream adds aging/fairness to LPM.
#
# Install: same .pth baked loader as sched_latch_fix (see ./Dockerfile); deferral
# via one-shot meta-path watcher on sglang.srt.managers.schedule_policy.
import importlib.util
import inspect
import os
import sys
import time

_TARGET = "sglang.srt.managers.schedule_policy"
_LOG_PREFIX = "[sched-lpm-waitfix]"


def _log(msg):
    print(f"{_LOG_PREFIX} {msg}", file=sys.stderr, flush=True)


def _env_params():
    try:
        t = float(os.environ.get("SGLANG_LPM_WAIT_BOOST_SECONDS", "0") or 0)
    except ValueError:
        t = 0.0
    try:
        n = int(os.environ.get("SGLANG_LPM_WAIT_BOOST_MAX", "1") or 1)
    except ValueError:
        n = 1
    return t, max(n, 1)


_state = {"boosts": 0, "warned": False}


def _boost(self, waiting_queue, lpm, t_seconds, max_n):
    try:
        if self.policy is not lpm or t_seconds <= 0 or len(waiting_queue) < 2:
            return
        now = time.perf_counter()
        overtime = []
        for r in waiting_queue:
            entry = getattr(getattr(r, "time_stats", None), "wait_queue_entry_time", 0.0)
            # 0.0 sentinel = never enqueued (or pre-init); retraction re-stamps,
            # which restarts its clock by design.
            if entry > 0.0 and now - entry > t_seconds:
                overtime.append((entry, id(r), r))
        if not overtime:
            return
        overtime.sort(key=lambda x: (x[0], x[1]))
        picks = {_id for _, _id, _ in overtime[:max_n]}
        chosen = [r for _, _id, r in overtime if _id in picks]
        waiting_queue[:] = chosen + [r for r in waiting_queue if id(r) not in picks]
        _state["boosts"] += 1
        if _state["boosts"] <= 3 or _state["boosts"] % 1000 == 0:
            waited = now - overtime[0][0]
            _log(f"boost #{_state['boosts']} n={len(chosen)} "
                 f"longest_wait={waited:.1f}s (T={t_seconds:.0f}s N={max_n})")
    except Exception as e:  # noqa: BLE001 — never break scheduling
        if not _state["warned"]:
            _state["warned"] = True
            _log(f"boost disabled after error: {type(e).__name__}: {e}")


def _install():
    from sglang.srt.managers import schedule_policy as sp

    if not hasattr(sp, "SchedulePolicy") or not hasattr(sp, "CacheAwarePolicy"):
        raise RuntimeError("drift: SchedulePolicy/CacheAwarePolicy not found in schedule_policy")
    lpm = getattr(sp.CacheAwarePolicy, "LPM", None)
    if lpm is None:
        raise RuntimeError("drift: CacheAwarePolicy.LPM missing")
    orig = sp.SchedulePolicy.calc_priority
    params = list(inspect.signature(orig).parameters)
    if "waiting_queue" not in params:
        raise RuntimeError(f"drift: calc_priority signature changed: {params}")

    t_seconds, max_n = _env_params()
    if t_seconds <= 0:
        _log("disabled (SGLANG_LPM_WAIT_BOOST_SECONDS unset/0) — pure LPM")
        return

    def calc_priority(self, waiting_queue, running_batch=None, *a, **k):
        orig(self, waiting_queue, running_batch, *a, **k)
        _boost(self, waiting_queue, lpm, t_seconds, max_n)

    sp.SchedulePolicy.calc_priority = calc_priority
    _log(f"installed (T={t_seconds:.0f}s N={max_n})")


class _DeferredArm:
    """One-shot meta-path watcher: patches SchedulePolicy right after
    schedule_policy finishes executing."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None
        real = spec.loader

        class _Wrapper:
            def create_module(self, s):
                return real.create_module(s) if hasattr(real, "create_module") else None

            def exec_module(self, s):
                real.exec_module(s)
                try:
                    _install()
                except Exception as e:  # noqa: BLE001
                    _log(f"INSTALL FAILED: {type(e).__name__}: {e}")

            def __getattr__(self, k):
                return getattr(real, k)

        spec.loader = _Wrapper()
        return spec


if not any(isinstance(f, _DeferredArm) for f in sys.meta_path):
    sys.meta_path.insert(0, _DeferredArm())
    _log("armed (waits for sglang.srt.managers.schedule_policy, then patches)")
