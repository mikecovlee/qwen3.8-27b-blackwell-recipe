#!/usr/bin/env python3
"""Local GSM8K eval against SGLang, 4-way concurrency.

Protocol mirrors the RadixArk qualification: thinking on, temp=1.0, top_p=0.95, top_k=20.
Usage: gsm8k.py <N> <out.json>
Requires a GSM8K test JSONL (one {"question","answer"} per line); path via GSM8K_DATA
(default ./gsm8k-test.jsonl). Download: huggingface.co/datasets/openai/gsm8k (test split).
"""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("SGLANG_BASE", "http://127.0.0.1:8080").rstrip("/") + "/v1/chat/completions"
MODEL = os.environ.get("MODEL_NAME", "qwen3.8-27b")
DATA = os.environ.get("GSM8K_DATA", "gsm8k-test.jsonl")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = sys.argv[2] if len(sys.argv) > 2 else "gsm8k-result.json"


def gold(ans):
    m = re.search(r"####\s*(.+)", ans)
    return m.group(1).strip().replace(",", "") if m else None


def extract(text):
    if not text:
        return None
    # prefer explicit #### marker
    m = re.findall(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return m[-1].replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


PROMPT = ("Solve the following math problem step by step. "
          "At the end, output the final numeric answer on its own line in the form '#### <number>'.\n\n"
          "Question: {q}")


def norm(x):
    if x is None:
        return None
    x = x.replace(",", "").rstrip(".").strip()
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return x


def call(q):
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": PROMPT.format(q=q)}],
               "temperature": 1.0, "top_p": 0.95, "top_k": 20,
               "max_tokens": 4096}
    req = urllib.request.Request(BASE, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    ch = r["choices"][0]["message"]
    return (ch.get("content") or ""), (ch.get("reasoning_content") or "")


def work(item):
    i, (q, g) = item
    t = time.time()
    try:
        content, reasoning = call(q)
        pred = extract(content) or extract(reasoning)
        ok = (norm(pred) == norm(g))
        return {"i": i, "gold": g, "pred": pred, "ok": ok, "wall": round(time.time() - t, 1),
                "content": (content or "")[-400:]}
    except Exception as e:
        return {"i": i, "gold": g, "pred": None, "ok": False, "err": str(e)[:120], "wall": round(time.time() - t, 1)}


def main():
    rows = [json.loads(l) for l in open(DATA)]
    rows = rows[:N]
    items = [(i, (r["question"], gold(r["answer"]))) for i, r in enumerate(rows)]
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for k, res in enumerate(ex.map(work, items), 1):
            results.append(res)
            if k % 10 == 0 or k == len(items):
                corr = sum(1 for r in results if r["ok"])
                el = time.time() - t0
                print(f"  {k}/{len(items)}  acc={corr}/{k}={corr/k:.3f}  elapsed={el:.0f}s  eta={el/k*(len(items)-k):.0f}s", flush=True)
    corr = sum(1 for r in results if r["ok"])
    acc = corr / len(results)
    summary = {"n": len(results), "correct": corr, "accuracy": acc,
               "protocol": "thinking, temp=1.0, top_p=0.95, top_k=20",
               "wall_s": round(time.time() - t0, 1), "results": results}
    json.dump(summary, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n== GSM8K local (n={len(results)}) ==  accuracy={acc:.4f} ({corr}/{len(results)})  wall={time.time()-t0:.0f}s")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
