#!/usr/bin/env python3
"""RULER multi-needle NIAH at ONE long length — completion protocol, fixed depths for cache reuse.

Usage: ruler-niah-multi.py <N> <out.json> [length] [tasks csv]
Env: SGLANG_BASE, MODEL_NAME, RULER_HAYSTACK (default ./haystack.txt).
"""
import json, os, random, re, sys, time, urllib.request
import tiktoken
from wonderwords import RandomWord

BASE = os.environ.get("SGLANG_BASE", "http://127.0.0.1:8080").rstrip("/")
MODEL = os.environ.get("MODEL_NAME", "qwen3.8-27b")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = sys.argv[2] if len(sys.argv) > 2 else "ruler-niah-multi.json"
LENGTH = int(sys.argv[3]) if len(sys.argv) > 3 else 240000
WANT = sys.argv[4].split(",") if len(sys.argv) > 4 else ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multivalue"]

ENC = tiktoken.get_encoding("cl100k_base")
RW = RandomWord()

TEMPLATE = ("Some special magic numbers are hidden within the following text. Make sure to memorize it. "
            "I will quiz you about the numbers afterwards.\n{context}\n"
            "What are all the special magic numbers for {query} mentioned in the provided text?")
ANSWER_PREFIX = " The special magic numbers for {query} mentioned in the provided text are"
NEEDLE = "One of the special magic numbers for {key} is: {value}."

# name -> (num_key, num_value, num_query)
ALL_TASKS = {"niah_single_1": (1, 1, 1), "niah_single_2": (1, 2, 1), "niah_single_3": (1, 3, 1),
             "niah_multikey_1": (4, 1, 4), "niah_multivalue": (1, 4, 1), "niah_multiquery": (4, 1, 4)}
TASKS = {k: ALL_TASKS[k] for k in WANT}

HAY = open(os.environ.get("RULER_HAYSTACK", "haystack.txt"), encoding="utf-8").read()
SENTS = [s.strip() for s in re.split(r"(?<=[.!?])\s+", HAY) if len(s.strip()) > 40]


def rnum(n=7):
    return str(random.randint(10 ** (n - 1), 10 ** n - 1))


def rword():
    return f"{RW.word(include_parts_of_speech=['adjectives'])}-{RW.word(include_parts_of_speech=['nouns'])}"


def build(target, nk, nv, nq, seed):
    rng = random.Random(seed)
    keys = [rword() for _ in range(nk)]
    values = [[rnum() for _ in range(nv)] for _ in range(nk)]
    needles = [NEEDLE.format(key=keys[ki], value=v) for ki in range(nk) for v in values[ki]]
    ctx, toks, i = [], 0, 0
    while toks < target - 200:
        s = SENTS[i % len(SENTS)]
        ctx.append(s); toks += len(ENC.encode(s)) + 1; i += 1
        if i > len(SENTS) * 8:
            break
    # fixed, evenly spaced depths in 0.60..0.90 (deterministic -> prefix cache hits)
    m = len(needles)
    positions = sorted(int(len(ctx) * (0.60 + 0.30 * j / max(m - 1, 1))) for j in range(m))
    out = []
    for idx, s in enumerate(ctx):
        out.append(s)
        while positions and positions[0] == idx:
            out.append(needles.pop(0)); positions.pop(0)
    context = " ".join(out)
    qidx = rng.sample(range(nk), nq)
    queries = [keys[j] for j in qidx]
    answers = [values[j][k] for j in qidx for k in range(nv)]
    query = ", ".join(queries[:-1]) + ", and " + queries[-1] if len(queries) > 1 else queries[0]
    prompt = TEMPLATE.format(context=context, query=query) + ANSWER_PREFIX.format(query=query)
    return prompt, answers


def call(prompt):
    payload = {"model": MODEL, "prompt": prompt, "temperature": 0.0, "max_tokens": 128}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["choices"][0].get("text", "")


def main():
    print(f"RULER multi-needle @ {LENGTH} tokens (completion, fixed depths), n={N}, tasks={list(TASKS)}", flush=True)
    res = []
    for task, (nk, nv, nq) in TASKS.items():
        for s in range(N):
            prompt, answers = build(LENGTH, nk, nv, nq, seed=hash((task, LENGTH, s)) & 0xFFFFFFFF)
            t = time.time()
            try:
                out = call(prompt)
                hit = sum(1.0 for a in answers if a.lower() in out.lower()) / len(answers)
                rec = {"task": task, "sample": s, "n_values": len(answers), "score": hit,
                       "out": out[:200], "wall": round(time.time() - t, 1)}
            except Exception as e:
                rec = {"task": task, "sample": s, "score": 0.0, "err": str(e)[:200], "wall": round(time.time() - t, 1)}
            res.append(rec)
            print(f"  {task:18s} s{s} score={rec['score']:.2f} wall={rec['wall']}s "
                  f"{rec.get('err','')[:70]} out={rec.get('out','')[:70]!r}", flush=True)
    agg = {}
    for r in res:
        agg.setdefault(r["task"], []).append(r["score"])
    print(f"\n== RULER @ {LENGTH//1024}K ==")
    for task in TASKS:
        print(f"  {task:18s} {sum(agg[task])/len(agg[task])*100:6.1f}%")
    json.dump({"length": LENGTH, "n": N, "results": res,
               "aggregate": {t: sum(v) / len(v) * 100 for t, v in agg.items()}}, open(OUT, "w"), indent=2)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
