#!/usr/bin/env python3
"""Fast RULER NIAH (single needle) — completion protocol, fixed depth=90% for prefix-cache reuse.

Usage: ruler-niah.py <N> <out.json> [lengths csv]
Env: SGLANG_BASE, MODEL_NAME, RULER_HAYSTACK (default ./haystack.txt).
"""
import json, os, random, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import tiktoken
from wonderwords import RandomWord

BASE = os.environ.get("SGLANG_BASE", "http://127.0.0.1:8080").rstrip("/")
MODEL = os.environ.get("MODEL_NAME", "qwen3.8-27b")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = sys.argv[2] if len(sys.argv) > 2 else "ruler-niah.json"
LENGTHS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [32768, 65536, 131072, 240000]
DEPTH = 0.9

ENC = tiktoken.get_encoding("cl100k_base")
RW = RandomWord()

TEMPLATE = ("A special magic number is hidden within the following text. Make sure to memorize it. "
            "I will quiz you about the number afterwards.\n{context}\n"
            "What is the special magic number mentioned in the provided text?")
ANSWER_PREFIX = " The special magic number mentioned in the provided text is"
NEEDLE = "One of the special magic numbers is: {value}."

HAY = open(os.environ.get("RULER_HAYSTACK", "haystack.txt"), encoding="utf-8").read()
SENTS = [s.strip() for s in re.split(r"(?<=[.!?])\s+", HAY) if len(s.strip()) > 40]


def build(target, seed):
    rng = random.Random(seed)
    val = str(rng.randint(10 ** 6, 10 ** 7 - 1))
    ctx, toks, i = [], 0, 0
    while toks < target - 100:
        s = SENTS[i % len(SENTS)]
        ctx.append(s); toks += len(ENC.encode(s)) + 1; i += 1
        if i > len(SENTS) * 8:
            break
    pos = int(len(ctx) * DEPTH)
    ctx.insert(pos, NEEDLE.format(value=val))
    context = " ".join(ctx)
    prompt = TEMPLATE.format(context=context) + ANSWER_PREFIX
    return prompt, val


def call(prompt):
    payload = {"model": MODEL, "prompt": prompt, "temperature": 0.0, "max_tokens": 32, "stop": ["\n"]}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["choices"][0].get("text", "")


def work(job):
    L, s = job
    prompt, val = build(L, seed=hash((L, s)) & 0xFFFFFFFF)
    t = time.time()
    try:
        out = call(prompt)
        return {"len": L, "sample": s, "hit": val in out, "val": val, "out": out[:120],
                "wall": round(time.time() - t, 1)}
    except Exception as e:
        return {"len": L, "sample": s, "hit": False, "err": str(e)[:160], "wall": round(time.time() - t, 1)}


def main():
    jobs = [(L, s) for L in LENGTHS for s in range(N)]
    print(f"RULER NIAH single needle, completion, depth={DEPTH:.0%}, n={N}, lengths={LENGTHS}", flush=True)
    res, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=1) as ex:
        for r in ex.map(work, jobs):
            res.append(r)
            print(f"  {r['len']//1024:>4d}K s{r['sample']} hit={r['hit']} wall={r['wall']}s out={r.get('out','')[:70]!r}", flush=True)
    print("\n== RULER NIAH single (accuracy) ==")
    for L in LENGTHS:
        rr = [r for r in res if r["len"] == L]
        print(f"  {L//1024:>4d}K  {sum(r['hit'] for r in rr)}/{len(rr)}")
    json.dump({"depth": DEPTH, "n": N, "results": res}, open(OUT, "w"), indent=2)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
