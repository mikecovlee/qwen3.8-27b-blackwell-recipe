# sched-latch-fix: 修复 sglang 5f55db35 树调度器假闩锁(mrr-1 并发墙)
#
# Bug(scheduler.py:3355):`len(adder.can_run_list) >= get_num_allocatable_reqs()`
#   把 chunked prefill 续传(已持有行、不申请新行)计入 can_run,与按空闲行数
#   算出的额度比较 → 双重计数 → batch_is_full 假闩锁;闩锁只在流 finish 时复位
#   → 最后一路请求被锁死(实测 mrr=N 只能跑 N-1)。
# Fix:仅当本 pass 真正的新准入数 < pass 起始空闲行数时,抑制来自 3355 的置位。
#   真容量耗尽(新准入已吃满空闲行)时闩锁照常,防 alloc_req_slots fail-loud。
# 挂载:-v .../sched-latch-fix:/patches:ro  -e PYTHONPATH=/patches
# 验证:evidence/README.md「Scheduler false-latch fix / 调度器假闩锁修复」节
import sys
import traceback


def _log(msg):
    print(f"[sched-latch-fix] {msg}", file=sys.stderr, flush=True)


_state = {"new_admits": 0, "rows_free": 0, "suppressed": 0, "attempts": 0}


def _install():
    from sglang.srt.managers import schedule_policy as sp
    from sglang.srt.managers import scheduler as sch
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    orig_gnb = sch.Scheduler._get_new_batch_prefill_raw

    def gnb(self, *a, **k):
        # 每 pass 一次:快照行余额、清零新准入计数
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
            if (caller is not None and caller.lineno == 3355
                    and caller.name == "_get_new_batch_prefill_raw"):
                _state["attempts"] += 1
                if _state["new_admits"] < _state["rows_free"]:
                    # 假闩锁:can_run 里的续传不占新行,行余额仍够 → 抑制
                    _state["suppressed"] += 1
                    if _state["suppressed"] <= 3 or _state["suppressed"] % 1000 == 0:
                        _log(f"suppressed false latch #{_state['suppressed']} "
                             f"(new_admits={_state['new_admits']} rows_free={_state['rows_free']})")
                    return
        self.__dict__["_bif_backing"] = v

    ScheduleBatch.batch_is_full = property(_bif_get, _bif_set)
    _log("installed")


try:
    _install()
except Exception as e:  # noqa: BLE001
    _log(f"INSTALL FAILED: {type(e).__name__}: {e}")
