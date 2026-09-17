#!/usr/bin/env python3
"""Acceptance: HiCache eviction → reload under pool overcommit (post-upgrade probe).

Oversubscribes the device KV pool with three cold sessions, then continues the
FIRST session verbatim and times it. The continuation must come back from the host
tier (seconds), not via full re-prefill (tens-to-hundreds of seconds), the engine
must log zero retractions/assertions throughout, and the honest host-load counter
(sglang:load_back_tokens_total{pool="kv"}) must show a positive delta — a fast
continuation without host load-back is device reuse, not an L2 reload.

Run against an isolated test instance (compose override with a different
container_name and host port — see verify-scheduling-patches.py docstring for the
two-line example):
  python3 verify-hicache-thrash.py --url http://localhost:8099/generate --container llm-infer-test

Sizing note: default prompts are tuned for a 262144-token pool (~0.9 token/char measured on
repetitive CJK). Pass --a-chars/--bc-chars scaled to your pool; total ≈ 1.4-1.6× pool
forces real eviction while staying inside the host tier (hicache ratio).
Also reports the tier evidence (#cached-token / #new-token of the last prefill) so a
mamba-state/KV-prefix mismatch (high hit score but full-replay latency) is visible.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080/generate")
ap.add_argument("--container", default="llm-infer")
ap.add_argument("--a-chars", type=int, default=244400)
ap.add_argument("--bc-chars", type=int, default=94600)
ap.add_argument("--reload-max-s", type=float, default=15.0)
ap.add_argument("--replay-s", type=float, default=30.0)
ARGS = ap.parse_args()


def load_back_kv_total():
    p = urlparse(ARGS.url)
    murl = f"{p.scheme}://{p.netloc}/metrics"
    with urllib.request.urlopen(murl, timeout=30) as r:
        text = r.read().decode()
    total = 0.0
    for line in text.splitlines():
        if line.startswith("sglang:load_back_tokens_total") and 'pool="kv"' in line:
            total += float(line.rsplit(" ", 1)[1])
    return total

T0 = time.time()
SALT = time.time()


def gen(text, max_tok=8):
    body = json.dumps({"text": text,
                       "sampling_params": {"max_new_tokens": max_tok, "temperature": 0},
                       "stream": True}).encode()
    req = urllib.request.Request(ARGS.url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    try:
        # Drain to the end: closing early aborts the request server-side, and an
        # aborted request's prefix is not guaranteed to reach the radix cache —
        # the eviction pressure and the continuation's cache both depend on it.
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                if first is None and line.startswith(b"data:") and b'"text"' in line:
                    first = time.time() - t0
    except Exception as e:
        print(f"request failed: {e}", file=sys.stderr)
    return first


unit = ("真hicache挤兑底稿" + str(SALT) + "。")
baseA = unit * (ARGS.a_chars // len(unit) + 1)
uB = ("挤兑方乙" + str(SALT) + "。")
uC = ("挤兑方丙" + str(SALT) + "。")

tA = gen(baseA)
print(f"A({ARGS.a_chars}c) cold prefill TTFT={tA:.0f}s" if tA else "A 失败")
tB = gen(uB * (ARGS.bc_chars // len(uB) + 1))
tC = gen(uC * (ARGS.bc_chars // len(uC) + 1))
print(f"B TTFT={tB:.0f}s  C TTFT={tC:.0f}s" if (tB is not None and tC is not None) else f"B/C 失败(B={tB}, C={tC})")
time.sleep(3)
try:
    lb_before = load_back_kv_total()
except Exception as e:
    lb_before = None
    print(f"metrics 不可达: {e}", file=sys.stderr)
tD = gen(baseA + "请回答:口令是什么?")
try:
    lb_after = load_back_kv_total()
except Exception as e:
    lb_after = None
    print(f"metrics 不可达: {e}", file=sys.stderr)
kv_delta = None if lb_before is None or lb_after is None else lb_after - lb_before
if kv_delta is not None:
    print(f"load_back kv: {lb_before:.0f} -> {lb_after:.0f} (+{kv_delta:.0f})")
print(f"D = A 的续传 TTFT={tD:.1f}s" if tD is not None else "D 失败")

p = subprocess.run(["docker", "logs", "--since", "15m", ARGS.container],
                   capture_output=True, text=True)
s = (p.stdout or "") + (p.stderr or "")
# The startup server_args dump is one giant line containing 'retraction_policy'
# (and could contain the word assert in an enum name); it is not an event — skip it.
retr_lines = [l for l in s.splitlines()
              if "retract" in l.lower() and "server_args=" not in l]
retr = len(retr_lines)
bad = sum(("assert" in l.lower() or "exception" in l.lower()) and "server_args=" not in l
          for l in s.splitlines())
if retr_lines:
    print("retract 行:")
    for l in retr_lines[-3:]:
        print("  ", l[:190])
pref = [l for l in s.splitlines() if "Prefill batch" in l]
if pref:
    print("tier 证据:", pref[-1][:190])
print(f"retract={retr} assert/exception={bad}")

if tD is None:
    verdict = "FAIL (D 请求失败)"
elif tD <= ARGS.reload_max_s and kv_delta is not None and kv_delta > 0 and retr == 0 and bad == 0:
    verdict = "PASS — host 层秒级重载, load_back kv 增量>0, 零 retraction/异常"
elif tD <= ARGS.reload_max_s and kv_delta == 0:
    verdict = "FAIL (快但无 host 装载: load_back kv 无增量, 疑似仅设备复用)"
elif kv_delta is None:
    verdict = "FAIL (metrics 不可达, 无法验证 host 装载)"
elif tD > ARGS.replay_s:
    verdict = f"REPLAY — 走了全量重放({tD:.0f}s):mamba/KV 错位或 host 层未覆盖,立案调查"
else:
    verdict = f"中间态({tD:.0f}s)—人工核对 tier 证据"
print("RESULT:", verdict)
sys.exit(0 if verdict.startswith("PASS") else 1)
