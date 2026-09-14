#!/usr/bin/env python3
"""Acceptance: scheduler patches under --schedule-policy lpm on the d6e72886 build.

Verifies both baked patches live, on a real load — designed to be run against an
ISOLATED test instance so production traffic never skews the timings. Bring one up
with a two-line compose override alongside your deployment file:

    # test-override.yml
    services:
      sglang:
        container_name: llm-infer-test
        ports: !override
          - "8099:8080"

    docker compose -f docker-compose.yml -f test-override.yml up -d
    python3 verify-scheduling-patches.py --url http://localhost:8099/generate --container llm-infer-test

T1 sched-latch-fix (false latch, scheduler.py:3661 anchor):
  A big cold prompt forces a long chunked prefill; a small request B is sent ~2.5s
  later. The bug latches batch_is_full until someone FINISHES, so the discriminator
  is NOT B's absolute TTFT — chunked prefill is inherently a single pipeline and B
  must queue behind A's prefill regardless. PASS := B starts decoding around A's
  PREFILL END, well before A's whole turn finishes.
T2 sched-lpm-waitfix (T=20s N=1 boost):
  two forced-long decodes (ignore_eos) fill both slots; cold C enqueues first, then
  hot continuations A2/B2 (~99% prefix hit) overtake it in LPM order. PASS := C is
  admitted before both hot overtakers (behavioral proof) with the boost hook active
  in this window — a fresh log line, or silence because the #1-#3 print quota was
  already spent before this test (see note below).
  Note: boost lines print only for events #1-#3 then every #1000 (throttle by
  design) — do not count log lines as the event counter.

Exit 0 only if both pass.
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
ap.add_argument("--boost-t", type=float, default=20.0, help="must match SGLANG_LPM_WAIT_BOOST_SECONDS")
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
    return (sum("suppressed false latch" in l for l in s.splitlines()),
            sum("waitfix] boost" in l for l in s.splitlines()))


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

# ---------------- T2: LPM wait boost ----------------
c1 = patch_counts()
o2 = {}
hb = ("热循环共享前缀" + str(SALT) + "。")
hs = []
for nm, rep, nt in (("H1", 300, 1024), ("H2", 310, 1600)):
    t = threading.Thread(target=fire, args=(o2, nm, hb * rep, nt))
    hs.append(t)
    t.start()
    time.sleep(0.4)
time.sleep(2.0)
tc = threading.Thread(target=fire, args=(o2, "Ccold", "全新冷会话内容" + str(SALT) + "。" * 3, 8))
tc.start()
time.sleep(1.0)
ta2 = threading.Thread(target=fire, args=(o2, "A2hot", hb * 300 + "续", 1024))
ta2.start()
time.sleep(1.0)
tb2 = threading.Thread(target=fire, args=(o2, "B2hot", hb * 310 + "续", 1024))
tb2.start()
for t in hs + [tc, ta2, tb2]:
    t.join(900)
c2 = patch_counts()
cg, a2g, b2g = o2.get("Ccold@g"), o2.get("A2hot@g"), o2.get("B2hot@g")
boost_logged = (c2[1] - c1[1]) >= 1
boost_throttled = c1[1] >= 3  # first-3 print quota already spent before T2 → silence expected
t2 = all(isinstance(x, float) for x in (cg, a2g, b2g)) and cg < a2g and cg < b2g \
    and (boost_logged or boost_throttled)
print(f"T2 boost:  C@{cg} A2@{a2g} B2@{b2g} boost-log {c1[1]}→{c2[1]}"
      f"{' (throttled)' if boost_throttled and not boost_logged else ''}")
print(f"           C TTFT={o2.get('Ccold_ttft')}s (无 boost 预期≈2回合, 有 boost 预期≤T{ARGS.boost_t:.0f}+1回合)")

print(f"\nlatch-log {c0[0]}→{c2[0]} | RESULT:", "PASS" if (t1 and t2) else "FAIL")
sys.exit(0 if (t1 and t2) else 1)
