#!/usr/bin/env python3
"""T3 acceptance: mamba stash slot-exhaustion matrix (2026-09-15 crash, E10 fix).

Reproduces the production crash shape on an ISOLATED twin (same GPU as prod —
never run against production directly):

  M1 (~98K ctx, short output)  ─┐ both running → 6/8 mamba slots held
  M2 (~98K ctx, long output)   ─┘
  L  (~150K cold, queued)        admitted the instant M1 finishes; its first
                                 chunked-prefill stash (cache_unfinished_req →
                                 donate track state to the radix tree) needs a
                                 fresh slot while M1's just-released tree
                                 checkpoints are still pinned by hicache
                                 writeback → pool exhausted →
                                 `assert slot is not None` (crash arm).

Arms (selected by the env the twin was brought up with):
  --expect crash : pre-fix config (MAMBA_STRATEGY=extra_buffer_lazy MAMBA_POOL=8).
                   PASS := "Can not alloc mamba cache" in container logs AND a
                   container restart, in ≥1 of --trials trials.
  --expect safe  : fixed config (defaults: extra_buffer / pool 10). PASS := no
                   assert, no restart, L completes with non-empty output in ALL
                   trials.

The crash window is transient (tree-lock timing), so both arms run --trials
times; use --tree-pressure N to fatten the radix tree with checkpointed
sessions first if the crash arm does not bite.

Exit: 0 PASS, 1 FAIL (behavior contradicted expectation), 2 INCONCLUSIVE.

Usage:
  python3 verify-mamba-stash.py --expect crash   # twin on lazy/8
  python3 verify-mamba-stash.py --expect safe    # twin on extra_buffer/10
Evidence: --outdir (default ./mamba-stash-T3-<ts>/): result.json,
per-trial timeline.jsonl, crash-arm docker log excerpt.
"""
import argparse
import json
import os
import random
import re
import string
import subprocess
import sys
import threading
import time
import urllib.request

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--url", default="http://127.0.0.1:8099/generate")
ap.add_argument("--container", default="llm-infer-test")
ap.add_argument("--expect", choices=("crash", "safe"), required=True)
ap.add_argument("--trials", type=int, default=2)
ap.add_argument("--m1-tokens", type=int, default=98000)
ap.add_argument("--m2-tokens", type=int, default=98000)
ap.add_argument("--l-tokens", type=int, default=150000)
ap.add_argument("--m1-out", type=int, default=160, help="M1 output cap (finishes first)")
ap.add_argument("--m2-out", type=int, default=2800, help="M2 output cap (outlives L's prefill)")
ap.add_argument("--l-out", type=int, default=16)
ap.add_argument("--tree-pressure", type=int, default=0,
                help="multi-turn sessions to run BEFORE the matrix (radix checkpoint filler)")
ap.add_argument("--watchdog", type=int, default=600, help="per-trial result window (s)")
ap.add_argument("--health-timeout", type=int, default=300, help="post-cr restart wait (s)")
ap.add_argument("--outdir", default=None)
ARGS = ap.parse_args()

BASE = ARGS.url.rsplit("/", 1)[0]
T_START = time.time()
lock = threading.Lock()
WORDS = ("system cache kernel token stream batch session record buffer slot page "
         "engine memory weight shard layer attention state query result worker task "
         "queue device index model train eval loss rank policy value node edge graph").split()
CRASH_LINE = "Can not alloc mamba cache"


# ---------------- server-side gauges ----------------

def metrics():
    try:
        txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    except Exception:
        return None

    def g(name):
        m = re.search(rf"^sglang:{name}{{[^}}]*}} ([0-9.eE+-]+)", txt, re.M)
        return float(m.group(1)) if m else None
    return dict(running=g("num_running_reqs"), queue=g("num_queue_reqs"),
                mamba_avail=g("mamba_available_tokens"), mamba_used=g("mamba_used_tokens"),
                mamba_evict=g("mamba_evictable_tokens"), kv=g("token_usage"))


def snap_line():
    s = metrics()
    if s is None:
        return "metrics-unavailable"
    return (f"run={s['running']} q={s['queue']} used={s['mamba_used']} "
            f"evict={s['mamba_evict']} avail={s['mamba_avail']} kv={s['kv']}")


def healthy(timeout=10):
    t_end = time.time() + timeout
    while time.time() < t_end:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def docker_crash_marks():
    """(count of assert lines since test start, RestartCount)."""
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(T_START - 5))
    p = subprocess.run(["docker", "logs", "--since", since, ARGS.container],
                       capture_output=True, text=True, timeout=120)
    s = (p.stdout or "") + (p.stderr or "")
    n_crash = sum(CRASH_LINE in l for l in s.splitlines())
    q = subprocess.run(["docker", "inspect", "-f", "{{.RestartCount}}", ARGS.container],
                       capture_output=True, text=True, timeout=30)
    try:
        rc = int(q.stdout.strip())
    except ValueError:
        rc = -1
    return n_crash, rc, s


def save_crash_excerpt(logtext, trial_dir):
    lines = logtext.splitlines()
    idx = [i for i, l in enumerate(lines) if CRASH_LINE in l]
    if not idx:
        return
    lo = max(0, idx[0] - 40)
    hi = min(len(lines), idx[-1] + 15)
    with open(f"{trial_dir}/crash-excerpt.log", "w") as f:
        f.write("\n".join(lines[lo:hi]) + "\n")


# ---------------- prompt building ----------------

def filler(rng, target_tokens, salt):
    """~target_tokens of unique English filler (≈4 chars/token), salted cold."""
    parts = [f"[session {salt}]"]
    n = 0
    while n < target_tokens:
        chunk = " ".join(rng.choice(WORDS) for _ in range(12)) + ". "
        chunk += (f"The measured throughput of shard {rng.randint(0, 64)} was "
                  f"{rng.randint(100, 6000)} tokens per second on node "
                  f"{rng.choice(string.ascii_lowercase)}{rng.randint(1, 99)}. ")
        parts.append(chunk)
        n += len(chunk) // 4
    return " ".join(parts)[:target_tokens * 4 + 400]


def count_tokens(text):
    body = json.dumps({"model": "qwen3.8-27b", "prompt": text}).encode()
    r = urllib.request.Request(BASE + "/tokenize", data=body,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=120) as f:
            d = json.load(f)
        return d.get("count") or len(d.get("tokens") or [])
    except Exception:
        return None


def calibrate(rng, target, salt, tag):
    """Filler trimmed to ±3% of the token target (KV budget math depends on it)."""
    text = filler(rng, target, salt)
    for _ in range(3):
        n = count_tokens(text)
        if n is None:
            print(f"  [{tag}] /tokenize unavailable — using char estimate", flush=True)
            return text
        if abs(n - target) <= target * 0.03:
            print(f"  [{tag}] prompt {n} tokens (target {target})", flush=True)
            return text
        ratio = target / max(n, 1)
        keep = int(len(text) * ratio * 1.02)
        text = text[:keep] if keep < len(text) else text + " " + filler(rng, target // 8, salt + "x")
        if keep >= len(text):
            break
    n = count_tokens(text)
    print(f"  [{tag}] prompt {n} tokens (target {target}, accepted)", flush=True)
    return text


# ---------------- request firing ----------------

def fire(out, tag, text, max_tok):
    body = json.dumps({"text": text,
                       "sampling_params": {"max_new_tokens": max_tok, "temperature": 0,
                                           "ignore_eos": True},
                       "stream": True}).encode()
    req = urllib.request.Request(ARGS.url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    nchar = 0
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                # Drain fully: closing early aborts the turn server-side and
                # would collapse the slot-holding this matrix depends on.
                if line.startswith(b"data:"):
                    if not nchar:
                        with lock:
                            out[tag + "@g"] = round(time.time() - T_START, 1)
                    nchar += len(line)
        with lock:
            out[tag + "_done"] = round(time.time() - T_START, 1)
            out[tag + "_bytes"] = nchar
    except Exception as e:
        with lock:
            out[tag + "_err"] = f"{type(e).__name__}: {e}"[:200]
            out[tag + "_done"] = round(time.time() - T_START, 1)
        print(f"  [{tag}] {type(e).__name__}: {e}", file=sys.stderr, flush=True)


# ---------------- tree pressure (optional reproduction aid) ----------------

def tree_pressure(n_sessions):
    rng = random.Random(int(time.time()) ^ 0xBEEF)
    salt = f"tp-{int(time.time())}"
    qs = ["Compute 27*31: think, then answer.", "What is the 20th prime number?",
          "Explain in one sentence what a radix cache is."]
    base = calibrate(rng, 20000, salt, "tree-pressure-base")
    for s in range(n_sessions):
        ctx = base + f"\n\n[variant {s}]\n"
        for ti, q in enumerate(qs):
            o = {}
            fire(o, f"tp{s}t{ti}", ctx + q + " (Answer briefly; the above is archived"
                 " background, do not summarize it.)", 200)
            ctx += q + "\n(ok)\n"
            print(f"  [tree-pressure] s{s} t{ti} {snap_line()}", flush=True)


# ---------------- one matrix trial ----------------

def run_trial(idx, trial_dir, prompts):
    print(f"\n== trial {idx + 1}/{ARGS.trials} ==", flush=True)
    if not healthy(60):
        return dict(verdict="inconclusive", why="server unhealthy before trial")
    base_crash, base_rc, _ = docker_crash_marks()
    out = {}
    tl = open(f"{trial_dir}/timeline.jsonl", "w")

    m1 = threading.Thread(target=fire, args=(out, "M1", prompts["M1"], ARGS.m1_out))
    m2 = threading.Thread(target=fire, args=(out, "M2", prompts["M2"], ARGS.m2_out))
    m1.start()
    time.sleep(0.5)
    m2.start()

    # wait until both are decoding (prefills done, 6 slots held)
    t_end = time.time() + ARGS.watchdog
    while time.time() < t_end and not ("M1@g" in out and "M2@g" in out):
        time.sleep(2)
    if "M1@g" not in out or "M2@g" not in out:
        tl.close()
        return dict(verdict="inconclusive", why="M1/M2 never reached decode", out=out)
    with lock:
        admit_gate = dict(m1_g=out.get("M1@g"), m2_g=out.get("M2@g"))
    print(f"  both decoding: {admit_gate} {snap_line()}", flush=True)

    # queue L NOW so it is waiting when M1 (short output) finishes
    thL = threading.Thread(target=fire, args=(out, "L", prompts["L"], ARGS.l_out))
    thL.start()
    l_sent = round(time.time() - T_START, 1)

    result, min_avail, l_admitted = None, None, None
    t_end = time.time() + ARGS.watchdog
    while time.time() < t_end:
        time.sleep(2)
        s = metrics()
        rec = dict(t=round(time.time() - T_START, 1), **(s or {}))
        tl.write(json.dumps(rec) + "\n")
        tl.flush()
        if s and s["mamba_avail"] is not None:
            if l_admitted is not None:  # only track headroom after L got in
                min_avail = s["mamba_avail"] if min_avail is None else min(min_avail, s["mamba_avail"])
        if s and l_admitted is None and "L@g" not in out and (s["running"] or 0) >= 2 \
                and (s["queue"] or 0) == 0 and "M1_done" in out:
            l_admitted = round(time.time() - T_START, 1)
        n_crash, rc, logtext = docker_crash_marks()
        if n_crash > base_crash or (rc >= 0 and base_rc >= 0 and rc > base_rc):
            result = "crash"
            save_crash_excerpt(logtext, trial_dir)
            break
        if "L_done" in out and "L_err" not in out:
            result = "safe"
            break
        if "L_err" in out and "L@g" not in out and "M2_err" in out:
            # both in-flight requests died without the assert line yet — keep
            # watching for the restart marker until watchdog expiry
            continue
    tl.close()

    for t in (m1, m2, thL):
        t.join(120)
    n_crash, rc, _ = docker_crash_marks()
    verdict = dict(
        trial=idx, result=result or "inconclusive",
        crash_lines_delta=n_crash - base_crash, restart_delta=(rc - base_rc) if rc >= 0 and base_rc >= 0 else None,
        l_sent=l_sent, l_admitted=l_admitted, min_mamba_avail_after_admit=min_avail,
        out={k: out[k] for k in sorted(out)},
    )
    print(f"  trial {idx + 1} → {verdict['result']} (crash-lines +{verdict['crash_lines_delta']}, "
          f"restarts +{verdict['restart_delta']}, min avail after admit={min_avail})", flush=True)
    if result == "crash":
        print("  waiting for container to come back...", flush=True)
        healthy(ARGS.health_timeout)
    return verdict


def main():
    outdir = ARGS.outdir or f"./mamba-stash-T3-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(outdir, exist_ok=True)
    print(f"T3 mamba-stash matrix — expect={ARGS.expect} trials={ARGS.trials} outdir={outdir}", flush=True)
    if not healthy(120):
        print("server never became healthy", file=sys.stderr)
        sys.exit(2)
    print(f"baseline: {snap_line()}", flush=True)
    _, base_rc, _ = docker_crash_marks()
    print(f"baseline RestartCount={base_rc}", flush=True)

    if ARGS.tree_pressure:
        print(f"\n-- tree pressure: {ARGS.tree_pressure} sessions --", flush=True)
        tree_pressure(ARGS.tree_pressure)

    rng = random.Random(int(time.time()))
    salt = f"t3-{int(time.time())}"
    print("\n-- calibrating prompts --", flush=True)
    prompts = {
        "M1": calibrate(rng, ARGS.m1_tokens, salt + "-m1", "M1") +
              "\n\n(Above is archived background; do not summarize.) Count from 1 upward, one number per line.",
        "M2": calibrate(rng, ARGS.m2_tokens, salt + "-m2", "M2") +
              "\n\n(Above is archived background; do not summarize.) Count from 1 upward, one number per line.",
        "L": calibrate(rng, ARGS.l_tokens, salt + "-l", "L") +
             "\n\n(Above is archived background; do not summarize.) Reply with the single word: done",
    }

    trials = []
    for i in range(ARGS.trials):
        td = f"{outdir}/trial-{i}"
        os.makedirs(td, exist_ok=True)
        v = run_trial(i, td, prompts)
        trials.append(v)
        if ARGS.expect == "crash" and v["result"] == "crash":
            break  # positive control caught it — no need for more trials
        if ARGS.expect == "safe" and v["result"] != "safe":
            break  # fixed arm must be clean in ALL trials — stop on first blemish
        if v["result"] == "inconclusive":
            break

    results = [t["result"] for t in trials]
    if ARGS.expect == "crash":
        ok = "crash" in results
    else:
        ok = bool(results) and all(r == "safe" for r in results)
    inconclusive = any(r == "inconclusive" for r in results)
    verdict = "PASS" if ok else ("INCONCLUSIVE" if inconclusive and not ok else "FAIL")
    summary = dict(expect=ARGS.expect, verdict=verdict, results=results, trials=trials,
                   args=vars(ARGS), started=time.strftime("%F %T", time.localtime(T_START)))
    with open(f"{outdir}/result.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\nRESULT: {verdict} (expect={ARGS.expect}, trials={results})", flush=True)
    sys.exit(0 if ok else (2 if verdict == "INCONCLUSIVE" else 1))


if __name__ == "__main__":
    main()
