# sched-latch-fix — false-latch scheduler patch for
# lmsysorg/sglang:v0.5.22 pinned at index digest
#   sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae
#   (= source tree baa9bef, v0.5.22 line, built 2026-10-19)
#
# Same bug as the b1259f3e (v0.5.21) tree: scheduler.py gates new admits with
#   len(adder.can_run_list) >= get_num_allocatable_reqs(running_bs, ...)
# while can_run_list always contains the chunked-prefill continuation
# (add_chunked_req appends it unconditionally — still true in baa9bef) and the
# quota side already discounts its held rows. Double counting latches
# batch_is_full until the next finish: mrr=N degenerates to N-1 during long
# chunked prefills.
# v0.5.22 did NOT fix this double counting: it only added (a) a pre-loop
# latch for allocatable<=0 guarded by `self.chunked_req is None`
# (scheduler.py:3999-4003) and (b) the #41841/#41872 empty-batch latch fix
# (the NO_TOKEN branch now uses `len(adder.can_run_list) > 0 or
# not running_batch.is_empty()` at :4158). The gate this patch suppresses
# still exists verbatim at :4098-4103.
# Fix: suppress the latch only while genuinely-free rows remain.
#
# Anchor moved: scheduler.py:3976 (e00930c) -> scheduler.py:4103 (baa9bef).
# The gate is unchanged in shape; get_num_allocatable_reqs now min()s with
# pp_budget (identical for the non-PP single-GPU profile). The install
# VERIFIES its anchor against live source before installing and refuses
# (loudly) on drift. Candidate `running_batch.batch_is_full = True` lines in
# this tree: 4002 (pre-loop guard — NOT the anchor), 4103 (the gate —
# ANCHOR), 4108 (PD prefill mode — NOT the anchor).
#
# Install: baked into a derived image (see ./Dockerfile):
#   this module -> site-packages, plus a 99-sched-latch-fix.pth line
#   "import sched_latch_fix". It must NOT import sglang at .pth time (the
#   editable sglang finder .pth runs after ours), so installation is
#   DEFERRED via a one-shot meta-path watcher that fires when
#   sglang.srt.managers.scheduler is first imported.
# See ../README.md for the digest -> build map and the re-anchor procedure.
import importlib.util
import sys
import traceback

_TARGET = "sglang.srt.managers.scheduler"
_ANCHOR_FUNC = "_get_new_batch_prefill_raw"
_ANCHOR_LINENO = 4103
_ANCHOR_STMT = "running_batch.batch_is_full = True"


def _log(msg):
    print(f"[sched-latch-fix/e3a4f060] {msg}", file=sys.stderr, flush=True)


_state = {"new_admits": 0, "rows_free": 0, "suppressed": 0, "attempts": 0}


def _verify_anchor(sch):
    import inspect

    src_file = inspect.getsourcefile(sch)
    with open(src_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    got = lines[_ANCHOR_LINENO - 1].strip() if 0 < _ANCHOR_LINENO <= len(lines) else "<EOF>"
    if got != _ANCHOR_STMT:
        cands = [i + 1 for i, ln in enumerate(lines) if ln.strip() == _ANCHOR_STMT]
        raise RuntimeError(
            f"anchor drift: scheduler.py:{_ANCHOR_LINENO} is {got!r}, not {_ANCHOR_STMT!r}; "
            f"candidate lines in this tree: {cands} — re-anchor per ../README.md before use"
        )
    _log(f"anchor verified: {src_file}:{_ANCHOR_LINENO}")


def _install():
    from sglang.srt.managers import schedule_policy as sp
    from sglang.srt.managers import scheduler as sch
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    _verify_anchor(sch)

    orig_gnb = sch.Scheduler._get_new_batch_prefill_raw

    def gnb(self, *a, **k):
        # once per pass: snapshot free rows, reset new-admit counter
        _state["rows_free"] = self.req_to_token_pool.available_size()
        _state["new_admits"] = 0
        return orig_gnb(self, *a, **k)

    sch.Scheduler._get_new_batch_prefill_raw = gnb

    orig_aor = sp.PrefillAdder.add_one_req

    def aor(self, req, *a, **k):
        r = orig_aor(self, req, *a, **k)
        if "CONTINUE" in str(r):
            _state["new_admits"] += 1
        return r

    sp.PrefillAdder.add_one_req = aor

    def _bif_get(self):
        return self.__dict__.get("_bif_backing", False)

    def _bif_set(self, v):
        if v and not self.__dict__.get("_bif_backing", False):
            stack = traceback.extract_stack(limit=4)
            caller = stack[-2] if len(stack) >= 2 else None
            if (caller is not None and caller.lineno == _ANCHOR_LINENO
                    and caller.name == _ANCHOR_FUNC):
                _state["attempts"] += 1
                if _state["new_admits"] < _state["rows_free"]:
                    # false latch: continuations in can_run hold no new rows, quota still free
                    _state["suppressed"] += 1
                    if _state["suppressed"] <= 3 or _state["suppressed"] % 1000 == 0:
                        _log(f"suppressed false latch #{_state['suppressed']} "
                             f"(new_admits={_state['new_admits']} rows_free={_state['rows_free']})")
                    return
        self.__dict__["_bif_backing"] = v

    ScheduleBatch.batch_is_full = property(_bif_get, _bif_set)
    _log("installed")


class _DeferredArm:
    """One-shot meta-path watcher: wraps the real loader of _TARGET so _install()
    runs right after that module finishes executing, whatever import mechanism
    (import, from-import, importlib.import_module, spawn) triggered it."""

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
    _log("armed (waits for sglang.srt.managers.scheduler, then patches)")
