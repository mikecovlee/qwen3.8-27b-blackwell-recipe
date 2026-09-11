#!/usr/bin/env python3
"""SGLang benchmark: prompt-eval and decode throughput vs context length.

Env: SGLANG_BASE (default http://127.0.0.1:8080), MODEL_NAME (default qwen3.8-27b).
"""
import os
import time, json, uuid, sys
import requests

BASE = os.environ.get("SGLANG_BASE", "http://127.0.0.1:8080").rstrip("/") + "/v1"
MODEL = os.environ.get("MODEL_NAME", "qwen3.8-27b")

UNIT = ("The llama.cpp project provides a minimal C++ API for running large language models locally. "
        "It supports many quantization formats and can offload computation to GPUs through CUDA. "
        "Tensor parallelism splits the model across multiple graphics cards, trading some communication "
        "overhead for higher memory capacity and faster compute. KV cache quantization reduces memory "
        "pressure at the cost of small accuracy degradation, and modern builds expose these knobs through "
        "command-line flags. Server mode speaks the OpenAI-compatible protocol, so clients such as opencode "
        "can point their base URL at a local endpoint. When benchmarking, keep the prompt cache in mind, "
        "because repeated prefixes are evaluated once and then reused on later requests. Generation speed "
        "is dominated by memory bandwidth, while prompt evaluation is dominated by compute throughput. A "
        "single RTX A5000 delivers roughly nine hundred gigabytes per second of bandwidth, which yields "
        "about thirty tokens per second for a twenty-seven billion parameter model quantized to Q4. Adding "
        "a second card approximately doubles the aggregate bandwidth, but PCIe synchronization and smaller "
        "per-GPU layer slices usually cut the theoretical gain by twenty to forty percent. Context length is "
        "bounded by the product of cache size and quantization level; larger contexts need smaller caches or "
        "more video memory. The best measurement practice is to send several uncached prompts of varying "
        "length and report the median, since the first run can be warmer than later ones. This paragraph is "
        "designed to be repeated so that it produces a stable and predictable token count for benchmarking "
        "purposes.")


def est_tokens(text):
    return int(len(text) / 5.5)


def build_prompt(target_tokens, seed):
    text = f"Message-Id: {seed}\n\n"
    while est_tokens(text) < target_tokens:
        text += UNIT + "\n"
    return text


def run(prompt, max_tokens, label):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    ttft = None
    prompt_tokens = 0
    completion_tokens = 0
    delta_tokens = 0
    first = False
    finish_reason = None
    with requests.post(f"{BASE}/chat/completions", json=payload, stream=True, timeout=1200) as r:
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if not first:
                ttft = time.time() - t0
                first = True
            if obj.get("choices"):
                ch = obj["choices"][0]
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
                d = ch.get("delta", {})
                if d.get("content"):
                    delta_tokens += 1
            u = obj.get("usage")
            if u:
                prompt_tokens = u.get("prompt_tokens", 0)
                completion_tokens = u.get("completion_tokens", 0)
    total = time.time() - t0
    if ttft is None:
        ttft = total
    if completion_tokens == 0:
        completion_tokens = delta_tokens
    pps = prompt_tokens / ttft if ttft > 0 else 0
    gen_time = total - ttft if ttft else total
    tps = (completion_tokens - 1) / gen_time if gen_time > 0 else 0
    print(f"[{label}]")
    print(f"  prompt eval : n={prompt_tokens:<7d} {ttft*1000:8.1f} ms  -> {pps:6.1f} tok/s")
    print(f"  generation  : n={completion_tokens:<7d} {gen_time*1000:8.1f} ms  -> {tps:6.1f} tok/s  (finish={finish_reason})")
    print(f"  TTFT {ttft:.2f}s  total {total:.2f}s")
    sys.stdout.flush()
    return dict(prompt=prompt_tokens, completion=completion_tokens, ttft=ttft, total=total, pps=pps, tps=tps)


def main():
    results = {}
    print("== S1: prompt eval ~2K (uncached) ==")
    results["2k"] = run(build_prompt(2000, str(uuid.uuid4())), 16, "2K prompt + 16 gen")

    print("== S2: prompt eval ~4K (uncached) ==")
    results["4k"] = run(build_prompt(4000, str(uuid.uuid4())), 16, "4K prompt + 16 gen")

    print("== S3: generation 512 tokens @ short ctx (x2) ==")
    p3 = ("Write a long, detailed technical article about running local large language models on consumer "
          "GPUs. Cover quantization, KV cache, context length, benchmarking methodology, and practical tuning "
          "tips. Be thorough and write at least eight hundred words.")
    results["gen1"] = run(f"Message-Id: {uuid.uuid4()}\n\n{p3}", 512, "512 gen run 1")
    results["gen2"] = run(f"Message-Id: {uuid.uuid4()}\n\n{p3}", 512, "512 gen run 2")

    print("== S4: generation 256 tokens @ ~64K ==")
    results["64k"] = run(build_prompt(64000, str(uuid.uuid4())), 256, "64K prompt + 256 gen")

    print("== S5: generation 128 tokens @ ~128K ==")
    results["128k"] = run(build_prompt(128000, str(uuid.uuid4())), 128, "128K prompt + 128 gen")

    print("== S6: generation 128 tokens @ ~240K (pool 260K 上限内) ==")
    results["240k"] = run(build_prompt(240000, str(uuid.uuid4())), 128, "240K prompt + 128 gen")

    print("\n== Summary ==")
    print(f"  prompt eval: 2K -> {results['2k']['pps']:.1f}   4K -> {results['4k']['pps']:.1f} tok/s")
    print(f"  generation (decode) vs ctx:")
    for k, lab, clab in [("gen1", "run1", "~1K"), ("gen2", "run2", "~1K"),
                         ("64k", "S4", "64K"), ("128k", "S5", "128K"), ("240k", "S6", "240K")]:
        r = results[k]
        print(f"    ctx={clab:<5s}  {r['tps']:6.1f} tok/s   (prompt_n={r['prompt']}, gen_n={r['completion']})")
    print(f"  TTFT: 2K {results['2k']['ttft']:.2f}s  4K {results['4k']['ttft']:.2f}s  64K {results['64k']['ttft']:.2f}s  128K {results['128k']['ttft']:.2f}s  240K {results['240k']['ttft']:.2f}s")


if __name__ == "__main__":
    main()
