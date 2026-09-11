#!/usr/bin/env python3
"""Concurrency validation driver for the SGLang server.

Usage: concurrency-load.py <b2|t1|t2|t3|soak>
Each scenario prints stream lines, ENGINE lines (parsed from container logs) and GATE verdicts.

Env:
  SGLANG_BASE   default http://127.0.0.1:8080
  GW_BASE       default http://localhost:8088   (gateway, used by the t3 scenario)
  GW_KEY        gateway API key; if unset the gateway sub-test is skipped
  C4_CONTAINER  inference container name (default llm-infer)
  MODEL_NAME    default qwen3.8-27b
"""
import importlib.util
import json
import os
import random
import subprocess
import sys
import threading
import time
import uuid

import requests

ROOT = os.environ.get("LLM_SERVICE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("bsg", f"{ROOT}/tools/benchmark.py")
bsg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsg)

API = os.environ.get("SGLANG_BASE", "http://127.0.0.1:8080").rstrip("/") + "/v1/chat/completions"
GW = os.environ.get("GW_BASE", "http://localhost:8088").rstrip("/") + "/v1/chat/completions"
CONTAINER = os.environ.get("C4_CONTAINER", "llm-infer")
MODEL = os.environ.get("MODEL_NAME", "qwen3.8-27b")


def stream_once(url, prompt, mx, tag, out, key=None):
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": mx, "stream": True,
               "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.time()
    ttft = None
    usage = {}
    try:
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=1800) as r:
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    c = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if c.get("usage"):
                    usage = c["usage"]
                ch = (c.get("choices") or [{}])[0]
                delta = (ch.get("delta") or {})
                if (delta.get("content") or delta.get("reasoning_content")) and ttft is None:
                    ttft = time.time() - t0
        gen = usage.get("completion_tokens", 0)
        det = usage.get("prompt_tokens_details") or {}
        out[tag] = {"start": t0, "end": time.time(), "ttft": ttft, "gen": gen,
                    "prompt": usage.get("prompt_tokens", 0),
                    "cached": det.get("cached_tokens")}
    except Exception as e:  # noqa: BLE001
        out[tag] = {"start": t0, "end": time.time(), "error": f"{type(e).__name__}:{str(e)[:100]}"}


def engine_stats(t0, t1):
    """解析窗口内容器日志:decode 引擎吞吐、峰值并发、retract/assert 计数。"""
    try:
        p = subprocess.run(["docker", "logs", "--since", str(int(t0) - 1),
                            "--until", str(int(t1) + 2), CONTAINER],
                           capture_output=True, text=True, timeout=60)
        txt = p.stdout + p.stderr
    except Exception as e:  # noqa: BLE001
        return {"log_err": str(e)[:80]}
    low = txt.lower()
    by_bs = {}
    queuemax = 0
    for ln in txt.splitlines():
        if "#queue-req:" in ln:
            try:
                queuemax = max(queuemax, int(ln.split("#queue-req:")[1].split(",")[0]))
            except (ValueError, IndexError):
                pass
        if "Decode batch" not in ln or "gen throughput (token/s):" not in ln:
            continue
        try:
            nb = int(ln.split("#running-req:")[1].split(",")[0])
            gt = float(ln.split("gen throughput (token/s):")[1].split(",")[0])
            graph = "cuda graph: True" in ln
        except (ValueError, IndexError):
            continue
        d = by_bs.setdefault(nb, {"g": [], "e": 0})
        if graph:
            d["g"].append(gt)
        else:
            d["e"] += 1
    st = {}
    for nb, d in sorted(by_bs.items()):
        g = sorted(d["g"])
        st[nb] = (g[len(g) // 2] if g else 0.0, len(g), d["e"])
    gts = [m for m, n, e in st.values()]
    return {"gt_max": max(gts) if gts else 0, "gt_med": sorted(gts)[len(gts) // 2] if gts else 0,
            "peak_running": max(by_bs) if by_bs else 0, "peak_queue": queuemax, "by_bs": st,
            "retract": low.count("retract"), "assert": low.count("not enough space"),
            "fatal": sum(low.count(k) for k in ("illegal memory access", "cuda error", "traceback"))}


def run_concurrent(scen, jobs, mx):
    out = {}
    ths = [threading.Thread(target=stream_once, args=(API, pr, mx, tg, out)) for tg, pr in jobs]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = max(o["end"] for o in out.values()) - min(o["start"] for o in out.values())
    for tg in sorted(out):
        o = out[tg]
        if "error" in o:
            print(f"[{scen}] {tg}: ERROR {o['error']}", flush=True)
            continue
        dtps = o["gen"] / (o["end"] - o["start"] - o["ttft"]) if o["ttft"] and o["gen"] else 0
        print(f"[{scen}] {tg}: prompt={o['prompt']} gen={o['gen']} ttft={o['ttft']:.1f}s decode={dtps:.1f} tok/s", flush=True)
    ok = [o for o in out.values() if "error" not in o]
    gens = sum(o["gen"] for o in ok)
    dsum = sum(o["gen"] / (o["end"] - o["start"] - o["ttft"]) for o in ok if o["ttft"] and o["gen"])
    es = engine_stats(t0, time.time())
    print(f"[{scen}] SUMMARY: n_ok={len(ok)}/{len(jobs)} wall={wall:.1f}s gen_sum={gens} "
          f"decode_tps_sum~{dsum:.1f} | ENGINE peak_running={es.get('peak_running')} "
          f"peak_queue={es.get('peak_queue')} "
          f"retract={es.get('retract')} assert={es.get('assert')} fatal={es.get('fatal')}", flush=True)
    for nb, (m, n, e) in sorted((es.get("by_bs") or {}).items()):
        print(f"[{scen}] BS-MED bs={nb} med={m:.1f} n={n} eager={e}", flush=True)
    return out, es


def uniq_prompt(n):
    body = bsg.build_prompt(n, f"c4-{uuid.uuid4()}")
    return body + "\n\n请基于以上文档编号列出至少500个要点,以'1.'开头,持续输出不要收尾。"


def gate(name, cond, detail):
    print(f"GATE-{name}: {'PASS' if cond else 'FAIL'} ({detail})", flush=True)
    return cond


def decode_tps(o):
    if o.get("ttft") and o.get("gen"):
        return o["gen"] / max(o["end"] - o["start"] - o["ttft"], 1e-6)
    return 0.0


def bs_stat(es, nb):
    m, n, e = (es.get("by_bs") or {}).get(nb, (0.0, 0, 0))
    return m, n, e


def scen_b2():
    print("=== B2 基线: 2×48K mx2048 (并发2) ===", flush=True)
    out, es = run_concurrent("B2", [("s1", uniq_prompt(48000)), ("s2", uniq_prompt(48000))], 2048)
    m, n, e = bs_stat(es, 2)
    gate("B2-valid", n >= 5 and e == 0, f"bs2 graph decode lines={n} eager={e} med={m:.1f}")
    return out, es


def scen_t1():
    print("=== T1: 4×48K mx2048 (并发4) ===", flush=True)
    jobs = [(f"s{i}", uniq_prompt(48000)) for i in range(4)]
    out, es = run_concurrent("T1", jobs, 2048)
    m2, n2, _ = bs_stat(es, 2)
    m4, n4, e4 = bs_stat(es, 4)
    gate("T1-valid", n4 >= 5 and e4 == 0, f"bs4 graph decode lines={n4} eager={e4} med={m4:.1f}")
    if n4 >= 5 and m2 > 0:
        gate("T1-bs2vs4-intratest", m4 >= 1.35 * m2, f"bs4/bs2 med = {m4:.1f}/{m2:.1f} = x{m4 / m2:.2f}")
    return out, es


def scen_t2():
    print("=== T2 混合: 2×80K + 2×40K mx2048 (≈240K, 池92%) ===", flush=True)
    jobs = [("h1", uniq_prompt(80000)), ("h2", uniq_prompt(80000)),
            ("m1", uniq_prompt(40000)), ("m2", uniq_prompt(40000))]
    out, es = run_concurrent("T2", jobs, 2048)
    m4, n4, e4 = bs_stat(es, 4)
    gate("T2-valid", n4 >= 3, f"bs4 lines={n4} eager={e4} med={m4:.1f}")
    gate("T2-safety", es.get("retract", 1) == 0 and es.get("assert", 1) == 0 and es.get("fatal", 1) == 0,
         f"retract={es.get('retract')} assert={es.get('assert')} fatal={es.get('fatal')}")
    return out, es


def scen_t3():
    print("=== T3 单流回归 + 网关计费 ===", flush=True)
    ok = True
    out, _ = run_concurrent("T3a", [("s", uniq_prompt(1000))], 128)
    d1 = list(out.values())[0]
    ok &= gate("T3a-single-1K", "error" not in d1 and d1["gen"] > 100,
               f"decode {decode_tps(d1):.1f} tok/s (基线≈42, 允许±10%)")
    out, _ = run_concurrent("T3b", [("s", uniq_prompt(64000))], 128)
    d2 = list(out.values())[0]
    ok &= gate("T3b-single-64K", "error" not in d2 and d2["gen"] > 100,
               f"decode {decode_tps(d2):.1f} tok/s (基线≈38)")
    p = uniq_prompt(2000)
    out, _ = run_concurrent("T3c", [("s", p)], 8)
    r1 = list(out.values())[0]
    time.sleep(2)
    out, _ = run_concurrent("T3c2", [("s", p)], 8)
    r2 = list(out.values())[0]
    ok &= gate("T3c-cached-TTFT", r2.get("cached", 0) > 1800 and r2["ttft"] < 1.5,
               f"cached={r2.get('cached')} ttft={r2['ttft']:.2f}s (冷 {r1['ttft']:.1f}s)")
    # Gateway E2E (cache-report + pricing pass-through); skip if no key is configured.
    key = os.environ.get("GW_KEY")
    if not key:
        print("GATE-T3d-gateway: SKIPPED(set GW_KEY to enable)", flush=True)
        return ok
    g1 = {}
    stream_once(GW, p, 8, "g", g1, key=key)  # 冷(建立网关侧同文)
    time.sleep(2)
    g2 = {}
    stream_once(GW, p, 8, "g", g2, key=key)
    r = g2["g"]
    ok &= gate("T3d-gateway", "error" not in r and r.get("cached", 0) > 1800,
               f"gw1 prompt={g1['g'].get('prompt')} → gw2 prompt={r.get('prompt')} cached={r.get('cached')}")
    return ok


def scen_soak(seconds=600):
    print(f"=== SOAK {seconds}s: 随机 2-4 并发混合 ctx ===", flush=True)
    rng = random.Random(0xC4)
    pool = {n: uniq_prompt(n) for n in (8000, 32000)}
    total = errs = 0
    worst = (0, "")
    t_end = time.time() + seconds
    while time.time() < t_end:
        k = rng.randint(2, 4)
        jobs = []
        for i in range(k):
            if rng.random() < 0.3:
                pr = pool[rng.choice(list(pool))]
            else:
                pr = uniq_prompt(rng.choice([1000, 8000, 32000, 48000]))
            jobs.append((f"q{total + i}", pr))
        out = {}
        ths = [threading.Thread(target=stream_once, args=(API, pr, rng.randint(8, 96), tg, out)) for tg, pr in jobs]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        for tg, o in out.items():
            total += 1
            if "error" in o:
                errs += 1
                print(f"[SOAK] {tg}: ERROR {o['error']}", flush=True)
            elif o["end"] - o["start"] > worst[0]:
                worst = (o["end"] - o["start"], tg)
        time.sleep(4)
    st = engine_stats(time.time() - seconds - 5, time.time())
    print(f"[SOAK] SUMMARY: requests={total} errors={errs} worst_wall={worst[0]:.0f}s({worst[1]}) | "
          f"ENGINE gt_max={st.get('gt_max')} peak_running={st.get('peak_running')} peak_queue={st.get('peak_queue')} "
          f"retract={st.get('retract')} assert={st.get('assert')} fatal={st.get('fatal')}", flush=True)
    gate("SOAK", errs == 0 and st.get("assert", 0) == 0 and st.get("fatal", 0) == 0,
         f"errors={errs} assert={st.get('assert')} fatal={st.get('fatal')} retract={st.get('retract')}")
    return total, errs, st


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "b2":
        scen_b2()
    elif mode == "t1":
        scen_t1()
    elif mode == "t2":
        scen_t2()
    elif mode == "t3":
        scen_t3()
    elif mode == "soak":
        scen_soak()
    else:
        print("unknown mode", mode)
        sys.exit(2)
