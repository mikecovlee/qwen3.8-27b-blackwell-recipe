#!/usr/bin/env python3
"""Acceptance: sched-latch patch on the mainline build (v0.5.20 line, hrrn).

Verifies the baked false-latch patch live, on a real load — designed to be run
against an ISOLATED test instance so production traffic never skews the timings.
Bring one up with a two-line compose override alongside your deployment file:

    # test-override.yml
    services:
      sglang:
        container_name: llm-infer-test
        ports: !override
          - "8099:8080"

    docker compose -f docker-compose.yml -f test-override.yml up -d
    python3 verify-scheduling-patches.py --url http://localhost:8099/generate --container llm-infer-test

T1 sched-latch-fix (false latch; anchor scheduler.py:3661 on d6e72886,
scheduler.py:3887 on 06e4f2ed):
  A big cold prompt forces a long chunked prefill; a small request B is sent ~2.5s
  later. The bug latches batch_is_full until someone FINISHES, so the discriminator
  is NOT B's absolute TTFT — chunked prefill is inherently a single pipeline and B
  must queue behind A's prefill regardless. PASS := B starts decoding around A's
  PREFILL END, well before A's whole turn finishes.

The former T2 (sched-lpm-waitfix starvation test) was retired on 2026-09-19
together with the waitfix module, when upstream `--schedule-policy hrrn`
(aging-based) became the mainline starvation mitigation; both live on in git
history. Starvation behavior under hrrn is watched via production metrics,
not gated here.

Exit 0 only if T1 passes.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080/generate")
ap.add_argument("--container", default="llm-infer")
ARGS = ap.parse_args()

T0 = time.time()
SALT = time.time()
lock = threading.Lock()


def fire(out, tag, text, max_tok, delay=0.0):
    if delay:
        time.sleep(delay)
    body = json.dumps({"text": text,
                       "sampling_params": {"max_new_tokens": max_tok, "temperature": 0,
                                           "ignore_eos": True},
                       "stream": True}).encode()
    req = urllib.request.Request(ARGS.url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            got_first = False
            # Drain to the end: closing early makes the server ABORT the turn,
            # which would collapse the slot-holding this test depends on.
            for line in r:
                if not got_first and line.startswith(b"data:") and b'"text"' in line:
                    got_first = True
                    with lock:
                        out[tag + "@g"] = round(time.time() - T0, 1)
                        out[tag + "_ttft"] = round(time.time() - t0, 1)
        with lock:
            out[tag + "_done"] = round(time.time() - T0, 1)
    except Exception as e:
        with lock:
            out[tag] = f"ERR {e}"
        print(f"  [{tag}] {e}", file=sys.stderr)


def patch_counts():
    p = subprocess.run(["docker", "logs", ARGS.container], capture_output=True, text=True)
    s = (p.stdout or "") + (p.stderr or "")
    return sum("suppressed false latch" in l for l in s.splitlines())


c0 = patch_counts()

# ---------------- T1: false latch ----------------
o1 = {}
thA = threading.Thread(target=fire, args=(o1, "A", ("闩锁复验冷料" + str(SALT) + "。") * 9400, 1024))
thA.start()
fire(o1, "B", "请小结", 8, delay=2.5)
thA.join(900)
a_g, a_done, b_g = o1.get("A@g"), o1.get("A_done"), o1.get("B@g")
# B legal lower bound = A's prefill end (a_g); bug behavior = b_g ≈ a_done (full turn)
t1 = all(isinstance(x, float) for x in (a_g, a_done, b_g)) and b_g <= a_g + 8
sep = a_done - a_g if a_g and a_done else 0
print(f"T1 latch:  A decode@{a_g} A turn-done@{a_done} B decode@{b_g}"
      f"  → B {'BEFORE' if t1 else 'AT'} A's finish (回合长度 {sep:.0f}s 的分辨窗)")

c1 = patch_counts()
print(f"\nlatch-log {c0}→{c1} | RESULT:", "PASS" if t1 else "FAIL")
sys.exit(0 if t1 else 1)
