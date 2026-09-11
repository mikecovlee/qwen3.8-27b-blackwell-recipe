#!/usr/bin/env python3
"""Cache-report check: send the same long prompt 3 times; cached_tokens should be > 0 from run 2.

Env: SGLANG_BASE (default http://127.0.0.1:8080), MODEL_NAME (default qwen3.8-27b).
"""
import json
import os
import urllib.request

BASE = os.environ.get("SGLANG_BASE", "http://127.0.0.1:8080").rstrip("/")
MODEL = os.environ.get("MODEL_NAME", "qwen3.8-27b")

LONG = ("深度学习中的注意力机制是Transformer架构的核心组件,它允许模型在处理序列时"
        "同时关注所有位置的token,并通过查询-键-值的交互捕获长距离依赖关系。" * 60)
PROMPT = LONG + "\n\n请只回复两个字:收到"


def chat(url, key=None):
    req = urllib.request.Request(
        url,
        data=json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 16,
        }).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


if __name__ == "__main__":
    for i in (1, 2, 3):
        d = chat(f"{BASE}/v1/chat/completions")
        u = d["usage"]
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        print(f"run{i} prompt={u['prompt_tokens']} cached={cached}")
