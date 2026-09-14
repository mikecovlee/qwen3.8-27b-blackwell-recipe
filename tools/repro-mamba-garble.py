#!/usr/bin/env python3
"""mamba 槽饱和 -> think 乱码 复现实验 (2026-09-11 计划定稿版)

用法:
  python3 repro-mamba-garble.py [--exposure 600] [--workers 10] [--max-workers 14]
                                [--outdir DIR] [--filler FILE]
  Env: SGLANG_BASE(默认 http://127.0.0.1:8080)

阶段:
  0. 校准:启动 N 个"RM 替身"压力 worker(全新随机前缀、非流式、max_tokens 700),
     等待 mamba_available==0 且 num_queue>0;不够逐个加,上限 --max-workers。
  1. 暴露 (--exposure 秒):V1(长多轮 ~30K)与 V2(短多轮)交替提问,
     每轮记录 reasoning/content 全文、cached_tokens、服务端槽位快照;
     在线乱码判定(U+FFFD / 同字符连发 / 中英碎片密度)。
  2. 拆载并输出 summary。

产物: --outdir(默认 ./garble-repro-<ts>)/victim.jsonl pressure.jsonl events.jsonl summary.txt
直连 :8080,不经过网关(不污染计费)。
"""
import argparse
import json
import os
import random
import re
import string
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("SGLANG_BASE", "http://127.0.0.1:8080").rstrip("/")
CHAT = BASE + "/v1/chat/completions"
MODEL = "qwen3.8-27b"

# ---------- helpers ----------

def post(payload, timeout=900):
    body = json.dumps(payload).encode()
    r = urllib.request.Request(CHAT, data=body,
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
    return (f"run={s['running']} q={s['queue']} "
            f"mamba_used={s['mamba_used']} avail={s['mamba_avail']} "
            f"evict={s['mamba_evict']} mamba_usage={s['mamba_usage']} kv={s['kv_usage']}")


WORDS = ("system cache kernel token stream batch session record buffer slot page "
         "engine memory weight shard layer attention state query result worker task "
         "queue device index model train eval loss rank policy value node edge graph").split()


def rand_prefix(rng, n_tok):
    """全新随机短文档(~n_tok tokens),保证新 radix 路径。"""
    parts = []
    while sum(len(p) // 4 for p in parts) < n_tok:
        parts.append("".join(rng.choices(string.ascii_lowercase + string.digits,
                                         k=rng.randint(4, 10))))
        parts.append(rng.choice(WORDS))
        if rng.random() < 0.2:
            parts.append("记录" + str(rng.randint(0, 999)) + "号样本,评估其质量。")
    return " ".join(parts)[:n_tok * 5]


# ---------- 乱码检测 ----------

FFFD = "\ufffd"
FRAG = re.compile(r"[\u4e00-\u9fff][A-Za-z_]{2,}|[A-Za-z_]{2,}[\u4e00-\u9fff]")
RUN = re.compile(r"(.)\1{6,}")


def garble_flags(text):
    hits = []
    if not text:
        return hits
    n = len(text)
    if FFFD in text:
        hits.append(f"FFFD x{text.count(FFFD)}")
    m = RUN.findall(text)
    if m:
        hits.append(f"char-run x{len(m)}")
    frags = FRAG.findall(text)
    if len(frags) >= 12 and n > 500:
        hits.append(f"frag-density x{len(frags)}")
    return hits


# ---------- 压力 worker(RM 替身) ----------

class Load:
    def __init__(self, wid, outdir, stop):
        self.wid, self.out, self.stop = wid, open(outdir + "/pressure.jsonl", "a"), stop
        self.rng = random.Random(1000 + wid)

    def run(self):
        while not self.stop.is_set():
            n = self.rng.randint(20, 320)
            prompt = rand_prefix(self.rng, n)
            msgs = [{"role": "user",
                     "content": prompt + "\n\n请为这段文本给一个 0-10 的整数质量分,先思考再只输出数字。"}]
            t0 = time.time()
            try:
                d = post({"model": MODEL, "messages": msgs, "temperature": 1.0,
                          "max_tokens": 700})
                c = d["choices"][0]["message"].get("content") or ""
                rec = dict(w=self.wid, t=time.strftime("%H:%M:%S"),
                           pt=d["usage"].get("prompt_tokens"),
                           ct=d["usage"].get("completion_tokens"),
                           fr=d["choices"][0].get("finish_reason"),
                           wall=round(time.time() - t0, 1))
            except Exception as e:
                rec = dict(w=self.wid, t=time.strftime("%H:%M:%S"), err=str(e)[:120])
            self.out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.out.flush()


# ---------- victim ----------

class Victim(threading.Thread):
    def __init__(self, name, base, qs, gap_lo, gap_hi, exposure, outdir, stop):
        super().__init__(daemon=True)
        self.name, self.base, self.qs = name, base, qs
        self.gap_lo, self.gap_hi = gap_lo, gap_hi
        self.deadline = time.time() + exposure
        self.stop = stop
        self.f = open(f"{outdir}/victim.jsonl", "a")
        self.ef = open(f"{outdir}/events.jsonl", "a")
        self.turn = 0
        self.hits = 0
        self.msgs = [{"role": "user", "content": base + "\n(以上为存档背景,无需总结。)"}]

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
                det = (u.get("prompt_tokens_details") or {})
                rec = dict(v=self.name, turn=self.turn,
                           t=time.strftime("%H:%M:%S"),
                           pt=u.get("prompt_tokens"), ct=u.get("completion_tokens"),
                           cached=det.get("cached_tokens"),
                           fr=d["choices"][0].get("finish_reason"),
                           wall=round(time.time() - t0, 1),
                           snap_pre=pre, snap_post=metrics(),
                           reasoning=m.get("reasoning_content") or "",
                           content=m.get("content") or "")
            except Exception as e:
                rec = dict(v=self.name, turn=self.turn, err=str(e)[:200])
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.f.flush()
            hits = garble_flags(rec.get("reasoning", "")) + garble_flags(rec.get("content", ""))
            if hits:
                self.hits += 1
                ev = {k: rec.get(k) for k in
                      ("v", "turn", "t", "pt", "ct", "cached", "fr", "snap_pre")}
                ev["flags"] = hits
                ev["sample"] = (rec.get("reasoning") or rec.get("content") or "")[:400]
                self.ef.write(json.dumps(ev, ensure_ascii=False) + "\n")
                self.ef.flush()
                print(f"[{self.name}] >>> GARBLE turn{rec.get('turn')} {hits} "
                      f"{snap_line()}", flush=True)
            print(f"[{self.name}] t{self.turn} pt={rec.get('pt')} "
                  f"cached={rec.get('cached')} ct={rec.get('ct')} fr={rec.get('fr')} "
                  f"{snap_line()}", flush=True)
            # 乱码轮不回填历史(避免自喂),回填一律用干净短答案
            ans = rec.get("content") or "ok"
            self.msgs.append({"role": "assistant", "content": ans[:200]})
            self.msgs = [self.msgs[0]] + self.msgs[1:][-10:]  # 固定存档前缀+近期轮,长度不漂移
            while time.time() < self.deadline and not self.stop.is_set():
                time.sleep(2)
                if random.random() < 0.02:
                    print(f"[{self.name}] gap {snap_line()}", flush=True)
                    break
            time.sleep(random.uniform(self.gap_lo, self.gap_hi) / 2)


QS = [
    "计算 27*31,先想再答。",
    "第 20 个质数是多少?",
    "若 a=5,b=11,求 a*b-a",
    "'algorithm' 有几个字母?",
    "3 的 7 次方是多少?",
    "用一句话说明 radix cache 的作用。",
    "128 字节等于多少比特?",
    "sum(1..50) 是多少?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure", type=int, default=600)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-workers", type=int, default=14)
    ap.add_argument("--ramp-timeout", type=int, default=120)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--filler", default=None, help="text file used as prompt filler; built-in paragraph if unset")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = args.outdir or f"garble-repro-{ts}"
    os.makedirs(outdir, exist_ok=True)
    print("outdir:", outdir, flush=True)

    if args.filler:
        filler = open(args.filler, encoding="utf-8").read()
    else:
        filler = "长对话场景下调度器与显存池的联合压力测试语料,用于复现状态饱和时的输出劣化。" * 40
    base_long = (filler * 40)[:28000]   # ~29.7K tokens,对齐事发 victim 的 30796
    base_short = (filler * 4)[:5000]

    stop = threading.Event()
    threads = []

    def add_workers(n):
        for _ in range(n):
            t = threading.Thread(target=lambda: Load(len(threads), outdir, stop).run(),
                                 daemon=True)
            t.start()
            threads.append(t)

    add_workers(args.workers)
    t_end_ramp = time.time() + args.ramp_timeout
    ok = False
    while time.time() < t_end_ramp and len(threads) < args.max_workers:
        time.sleep(10)
        s = metrics()
        if not s:
            continue
        print(f"[ramp] {snap_line()} workers={len(threads)}", flush=True)
        if s["mamba_avail"] == 0 and s["queue"] > 0:
            ok = True
            break
        if (s["mamba_avail"] or 9) > 0:
            add_workers(2)
            print("[ramp] saturating more, workers ->", len(threads), flush=True)
    s = metrics()
    print("[ramp done] ok=", ok, snap_line(), flush=True)

    QS2 = ["27*31=?", "第 20 个质数?", "'algorithm' 几个字母?", "sum(1..50)?"]
    v1 = Victim("V1-long", base_long, QS, 15, 30, args.exposure, outdir, stop)
    v1b = Victim("V1b-branch", base_long, QS2, 15, 30, args.exposure, outdir, stop)  # 同前缀分叉,复刻事发结构
    v2 = Victim("V2-short", base_short, QS, 5, 15, args.exposure, outdir, stop)
    t0 = time.time()
    v1.start()
    v1b.start()
    v2.start()
    while (v1.is_alive() or v1b.is_alive() or v2.is_alive()):
        time.sleep(15)
        print(f"[watch] {int(time.time()-t0)}s {snap_line()}", flush=True)

    stop.set()
    time.sleep(5)
    summary = (f"exposure={int(time.time()-t0)}s ramp_ok={ok}\n"
               f"V1 turns={v1.turn} garble={v1.hits}\n"
               f"V1b turns={v1b.turn} garble={v1b.hits}\n"
               f"V2 turns={v2.turn} garble={v2.hits}\n"
               f"final {snap_line()}\n")
    open(outdir + "/summary.txt", "w").write(summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
