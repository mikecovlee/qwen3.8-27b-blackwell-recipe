#!/usr/bin/env python3
"""RULER freq_words_extraction (FWE) — coded-text frequency, completion protocol.
Faithful: synthetic 6-letter vocab (size=len//50), zeta(alpha=2) freq, answer=top3, metric=string_match_all.

Usage: ruler-fwe.py <N> <out.json> [lengths csv]
Env: SGLANG_BASE, MODEL_NAME.
"""
import json, os, random, string, sys, time, urllib.request
import tiktoken

BASE = os.environ.get("SGLANG_BASE", "http://127.0.0.1:8080").rstrip("/")
MODEL = os.environ.get("MODEL_NAME", "qwen3.8-27b")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = sys.argv[2] if len(sys.argv) > 2 else "ruler-fwe.json"
LENGTHS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [32768, 240000]
ALPHA = 2.0

ENC = tiktoken.get_encoding("cl100k_base")

TEMPLATE = ("Read the following coded text and track the frequency of each coded word. "
            "Find the three most frequently appeared coded words. {context}\n"
            "Question: Do not provide any explanation. Please ignore the dots '....'. "
            "What are the three most frequently appeared words in the above coded text?")
ANSWER_PREFIX = " Answer: According to the coded text above, the three most frequently appeared words are:"


def make_vocab(vsize, seed):
    rng = random.Random(seed)
    v = set()
    while len(v) < vsize:
        v.add(''.join(rng.choices(string.ascii_lowercase, k=6)))
    v = sorted(v)
    rng.shuffle(v)
    v[0] = '...'  # noise, highest rank
    return v


def zeta(a, V):
    return sum(k ** -a for k in range(1, V + 1))


def gen_text(vocab, num_words, seed):
    rng = random.Random(seed)
    V = len(vocab)
    z = zeta(ALPHA, V)
    words = []
    for i, w in enumerate(vocab, start=1):
        c = int(num_words * (i ** -ALPHA) / z)
        if c > 0:
            words.extend([w] * c)
    rng.shuffle(words)
    return ' '.join(words), vocab[1:4]


def build(target, seed):
    vsize = max(200, target // 50)
    vocab = make_vocab(vsize, seed)
    # calibrate num_words to hit ~target tokens
    n = int(target / 1.4)
    for _ in range(4):
        text, ans = gen_text(vocab, n, seed)
        tk = len(ENC.encode(text))
        if abs(tk - target) / target < 0.03:
            break
        n = max(100, int(n * target / max(tk, 1)))
    prompt = TEMPLATE.format(context=text) + ANSWER_PREFIX
    return prompt, ans


def call(prompt):
    payload = {"model": MODEL, "prompt": prompt, "temperature": 0.0, "max_tokens": 64}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["choices"][0].get("text", "")


def main():
    print(f"RULER FWE completion, n={N}, lengths={LENGTHS}", flush=True)
    res = []
    for L in LENGTHS:
        for s in range(N):
            t = time.time()
            prompt, ans = build(L, seed=hash((L, s)) & 0xFFFFFFFF)
            print(f"  built {L//1024}K s{s}: prompt~{len(ENC.encode(prompt))} tok in {time.time()-t:.1f}s; sending...", flush=True)
            t = time.time()
            try:
                out = call(prompt)
                hit = sum(1.0 for a in ans if a.lower() in out.lower()) / len(ans)
                rec = {"len": L, "sample": s, "score": hit, "ans": ans, "out": out[:200], "wall": round(time.time() - t, 1)}
            except Exception as e:
                rec = {"len": L, "sample": s, "score": 0.0, "err": str(e)[:200], "wall": round(time.time() - t, 1)}
            res.append(rec)
            print(f"    -> score={rec['score']:.2f} wall={rec['wall']}s {rec.get('err','')[:70]} out={rec.get('out','')[:100]!r}", flush=True)
    print("\n== RULER FWE ==")
    for L in LENGTHS:
        rr = [r for r in res if r["len"] == L]
        print(f"  {L//1024:>4d}K  {sum(r['score'] for r in rr)/len(rr)*100:6.1f}%  ({len(rr)} samples)")
    json.dump({"n": N, "lengths": LENGTHS, "results": res}, open(OUT, "w"), indent=2)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
