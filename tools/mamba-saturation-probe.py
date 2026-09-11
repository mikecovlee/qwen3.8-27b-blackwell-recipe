#!/usr/bin/env python3
"""mamba slot saturation + garble detector harness (investigation tool).

Reproduces the conditions of the 2026-09-11 incident on a hybrid (linear-attention)
model: an "RM scoring"-style pressure loop (fresh random prefix per request,
non-streaming, generation to a length cap) saturates max-running-requests and the
mamba state pool, while multi-turn victim sessions (a long shared-prefix pair that
mimics agent conversations, plus a short one) keep reusing and re-acquiring path
states. Every victim turn is scanned for token-soup garbling, with server-side
slot gauges snapshotted alongside.

Phases:
  0. ramp: start N pressure workers, wait until mamba available == 0 and queue > 0
     (add workers up to --max-workers).
  1. exposure (--exposure seconds): run victims; log turns to victim.jsonl,
     detections to events.jsonl.
  2. teardown: summary.

Env: SGLANG_BASE (default http://127.0.0.1:8080), MODEL_NAME (default qwen3.8-27b)
"""
import argparse, json, os, random, re, string, threading, time, urllib.request

BASE = os.environ.get("SGLANG_BASE", "http://127.0.0.1:8080").rstrip("/")
MODEL = os.environ.get("MODEL_NAME", "qwen3.8-27b")


def post(payload, timeout=900):
    body = json.dumps(payload).encode()
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.load(f)


def metrics():
    try:
        txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    except Exception:
        return None

    def g(name):
        m = re.search(rf"^sglang:{name}{{[^}}]*}} ([0-9.eE+-]+)", txt, re.M)
        return float(m.group(1)) if m else None
    return dict(running=g("num_running_reqs"), queue=g("num_queue_reqs"),
                mamba_usage=g("mamba_usage"), mamba_avail=g("mamba_available_tokens"),
                mamba_used=g("mamba_used_tokens"), mamba_evict=g("mamba_evictable_tokens"),
                kv_usage=g("token_usage"))


def snap_line():
    s = metrics()
    if s is None:
        return "metrics-unavailable"
    return (f"run={s['running']} q={s['queue']} used={s['mamba_used']} "
            f"evict={s['mamba_evict']} avail={s['mamba_avail']} "
            f"mamba_usage={s['mamba_usage']} kv={s['kv_usage']}")


WORDS = ("system cache kernel token stream batch session record buffer slot page "
         "engine memory weight shard layer attention state query result worker task "
         "queue device index model train eval loss rank policy value node edge graph").split()


def rand_doc(rng, n_tok):
    """Fresh random short 'document' (~n_tok tokens) to force new radix paths."""
    parts = []
    while sum(len(p) // 4 for p in parts) < n_tok:
        parts.append("".join(rng.choices(string.ascii_lowercase + string.digits,
                                         k=rng.randint(4, 10))))
        parts.append(rng.choice(WORDS))
        if rng.random() < 0.2:
            parts.append(f"sample-{rng.randint(0, 9999)} quality note {rng.randint(0, 99)}.")
    return " ".join(parts)[:n_tok * 5]


FFFD = "\ufffd"
FRAG = re.compile(r"[\u4e00-\u9fff][A-Za-z_]{2,}|[A-Za-z_]{2,}[\u4e00-\u9fff]")
RUN = re.compile(r"(.)\1{6,}")


def garble_flags(text):
    """Strict-ish detector; everything flagged is saved for manual review anyway."""
    hits = []
    if not text:
        return hits
    n = len(text)
    if FFFD in text:
        hits.append(f"FFFD x{text.count(FFFD)}")
    if RUN.search(text):
        hits.append("char-run")
    frags = FRAG.findall(text)
    if len(frags) >= 12 and n > 500:
        hits.append(f"frag-density x{len(frags)}")
    return hits


class Pressure(threading.Thread):
    """Stands in for an offline scoring loop: unique prompts, capped generations."""

    def __init__(self, wid, outdir, stop):
        super().__init__(daemon=True)
        self.wid, self.out, self.stop = wid, open(f"{outdir}/pressure.jsonl", "a"), stop
        self.rng = random.Random(1000 + wid)

    def run(self):
        while not self.stop.is_set():
            doc = rand_doc(self.rng, self.rng.randint(20, 320))
            msgs = [{"role": "user", "content": doc +
                     "\n\nGive a 0-10 integer quality score; think briefly, end with the number alone."}]
            t0 = time.time()
            try:
                d = post({"model": MODEL, "messages": msgs, "temperature": 1.0,
                          "max_tokens": 700})
                rec = dict(w=self.wid, t=time.strftime("%H:%M:%S"),
                           pt=d["usage"].get("prompt_tokens"),
                           ct=d["usage"].get("completion_tokens"),
                           fr=d["choices"][0].get("finish_reason"),
                           wall=round(time.time() - t0, 1))
            except Exception as e:
                rec = dict(w=self.wid, t=time.strftime("%H:%M:%S"), err=str(e)[:120])
            self.out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.out.flush()


QS = [
    "Compute 27*31: think, then answer.",
    "What is the 20th prime number?",
    "If a=5, b=11, compute a*b-a.",
    "How many letters are in the word 'algorithm'?",
    "What is 3 to the 7th power?",
    "Explain in one sentence what a radix cache is.",
    "How many bits are in 128 bytes?",
    "What is sum(1..50)?",
]


class Victim(threading.Thread):
    def __init__(self, name, base, qs, gap_lo, gap_hi, exposure, outdir, stop):
        super().__init__(daemon=True)
        self.name, self.qs = name, qs
        self.gap_lo, self.gap_hi = gap_lo, gap_hi
        self.deadline = time.time() + exposure
        self.stop = stop
        self.f = open(f"{outdir}/victim.jsonl", "a")
        self.ef = open(f"{outdir}/events.jsonl", "a")
        self.turn, self.hits = 0, 0
        self.msgs = [{"role": "user",
                      "content": base + "\n(The above is archived background; do not summarize.)"}]

    def run(self):
        qi = 0
        while time.time() < self.deadline and not self.stop.is_set():
            self.turn += 1
            q = self.qs[qi % len(self.qs)]
            qi += 1
            self.msgs.append({"role": "user", "content": q})
            pre = metrics()
            t0 = time.time()
            try:
                d = post({"model": MODEL, "messages": list(self.msgs),
                          "temperature": 0.6, "max_tokens": 400})
                m = d["choices"][0]["message"]
                u = d.get("usage", {})
                det = u.get("prompt_tokens_details") or {}
                rec = dict(v=self.name, turn=self.turn, t=time.strftime("%H:%M:%S"),
                           pt=u.get("prompt_tokens"), ct=u.get("completion_tokens"),
                           cached=det.get("cached_tokens"),
                           fr=d["choices"][0].get("finish_reason"),
                           wall=round(time.time() - t0, 1),
                           snap_pre=pre,
                           reasoning=m.get("reasoning_content") or "",
                           content=m.get("content") or "")
            except Exception as e:
                rec = dict(v=self.name, turn=self.turn, err=str(e)[:200])
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.f.flush()
            hits = garble_flags(rec.get("reasoning", "")) + garble_flags(rec.get("content", ""))
            if hits:
                self.hits += 1
                ev = {k: rec.get(k) for k in ("v", "turn", "t", "pt", "ct", "cached", "fr", "snap_pre")}
                ev["flags"] = hits
                ev["sample"] = (rec.get("reasoning") or rec.get("content") or "")[:400]
                self.ef.write(json.dumps(ev, ensure_ascii=False) + "\n")
                self.ef.flush()
                print(f"[{self.name}] >>> GARBLE turn{rec.get('turn')} {hits} {snap_line()}", flush=True)
            print(f"[{self.name}] t{self.turn} pt={rec.get('pt')} cached={rec.get('cached')} "
                  f"ct={rec.get('ct')} fr={rec.get('fr')} {snap_line()}", flush=True)
            # never feed suspected garbage back: assistant turn gets a clean stub
            self.msgs.append({"role": "assistant", "content": (rec.get("content") or "ok")[:200]})
            self.msgs = [self.msgs[0]] + self.msgs[1:][-10:]  # pinned base + recent turns, no drift
            time.sleep(random.uniform(self.gap_lo, self.gap_hi))


def filler_base(rng, target_tokens):
    parts = []
    while sum(len(p) // 4 for p in parts) < target_tokens:
        parts.append(" ".join(rng.choice(WORDS) for _ in range(12)) + ".")
        parts.append(f"The measured throughput of shard {rng.randint(0, 64)} was "
                     f"{rng.randint(100, 6000)} tokens per second on node "
                     f"{rng.choice(string.ascii_lowercase)}{rng.randint(1, 99)}.")
    return " ".join(parts)[:target_tokens * 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure", type=int, default=600)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-workers", type=int, default=14)
    ap.add_argument("--ramp-timeout", type=int, default=120)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or f"./mamba-probe-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(outdir, exist_ok=True)
    print("outdir:", outdir, flush=True)

    rng = random.Random(42)
    base_long = filler_base(rng, 28000)   # ~30K tokens, matches the incident victim size
    base_short = filler_base(rng, 2500)

    stop = threading.Event()
    workers = []

    def add_workers(n):
        for _ in range(n):
            t = Pressure(len(workers), outdir, stop)
            t.start()
            workers.append(t)

    add_workers(args.workers)
    t_end_ramp = time.time() + args.ramp_timeout
    ok = False
    while time.time() < t_end_ramp and len(workers) < args.max_workers:
        time.sleep(10)
        s = metrics()
        if not s:
            continue
        print(f"[ramp] {snap_line()} workers={len(workers)}", flush=True)
        if s["mamba_avail"] == 0 and s["queue"] > 0:
            ok = True
            break
        if (s["mamba_avail"] or 9) > 0:
            add_workers(2)
    print("[ramp done] ok=", ok, snap_line(), flush=True)

    v1 = Victim("V1-long", base_long, QS, 15, 30, args.exposure, outdir, stop)
    v1b = Victim("V1b-branch", base_long, QS, 15, 30, args.exposure, outdir, stop)
    v2 = Victim("V2-short", base_short, QS, 5, 15, args.exposure, outdir, stop)
    t0 = time.time()
    v1.start(); v1b.start(); v2.start()
    while v1.is_alive() or v1b.is_alive() or v2.is_alive():
        time.sleep(15)
        print(f"[watch] {int(time.time()-t0)}s {snap_line()}", flush=True)

    stop.set()
    time.sleep(5)
    summary = (f"exposure={int(time.time()-t0)}s ramp_ok={ok}\n"
               f"V1 turns={v1.turn} garble={v1.hits}\n"
               f"V1b turns={v1b.turn} garble={v1b.hits}\n"
               f"V2 turns={v2.turn} garble={v2.hits}\n"
               f"final {snap_line()}\n")
    open(f"{outdir}/summary.txt", "w").write(summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
