#!/usr/bin/env python3
"""Quality spot-check: run fixed prompts against an SGLang OpenAI endpoint.

Usage: quality-spotcheck.py <base_url> <model> <out.json>
Env fallbacks: SGLANG_BASE, MODEL_NAME. Deterministic (temperature=0) so FP8 vs FP4 outputs can be diffed.
"""
import json
import os
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SGLANG_BASE", "http://127.0.0.1:8080").rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MODEL_NAME", "qwen3.8-27b")
OUT = sys.argv[3] if len(sys.argv) > 3 else "quality-spotcheck.json"

SHORT = [
    ("math_mul", "计算 1234 × 5678,只输出最终数字。"),
    ("math_ratio", "一个班有 48 名学生,男生人数是女生的 3 倍。男生有多少人?只输出数字。"),
    ("squares", "输出 1 到 10 每个数的平方,用逗号分隔,只输出结果。"),
    ("logic_age", "父亲今年 40 岁,儿子 10 岁。几年后父亲的年龄是儿子的 2 倍?只输出数字。"),
    ("chicken_rabbit", "笼子里有鸡和兔共 35 个头、94 只脚。鸡有多少只?只输出数字。"),
    ("knowledge", "珠穆朗玛峰的海拔高度大约是多少米?只输出数字。"),
    ("translate", "把这句话翻译成英文,只输出译文:人工智能正在改变世界。"),
    ("json_out", '只输出一个 JSON 对象,包含字段 name(值"张三")和 age(值 30),不要多余文字。'),
    ("code_prime", "用 Python 写一个函数 is_prime(n) 判断素数,只输出代码块。"),
    ("reason_steps", "某数加 5 后乘以 3,再减去 12,结果是 33。这个数是多少?只输出数字。"),
    ("long_form", "用大约 150 字介绍光合作用,只输出正文。"),
    ("instr_fmt", "把单词 apple, banana, cherry 按字母顺序排列,用 | 分隔,只输出结果。"),
]

FILLER = (
    "在遥远的山谷里,有一座安静的图书馆,馆中收藏着各个年代的普通书籍,"
    "管理员每天整理书架,记录借阅情况,日子平淡而规律。"
)


def make_needle(n_repeat: int = 1800, needle: str = "739128"):
    """Build a long prompt (~n_repeat*len tokens) with a single needle."""
    body = (FILLER * n_repeat)
    mid = len(body) // 2
    body = body[:mid] + f"【重要】仓库的魔术编号是 {needle}。请记住它。" + body[mid:]
    q = "\n\n问题:上面文字里提到的魔术编号是多少?只输出数字。"
    return body + q


def call(prompt, max_tokens=600):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.time() - t
    msg = r["choices"][0]["message"]
    return {
        "content": msg.get("content", ""),
        "usage": r.get("usage"),
        "wall_s": round(dt, 2),
    }


def main():
    results = {"base": BASE, "model": MODEL, "short": {}, "needle": None}
    for name, prompt in SHORT:
        try:
            results["short"][name] = call(prompt)
            print(f"[short:{name}] {results['short'][name]['content'][:80]!r}", flush=True)
        except Exception as e:
            results["short"][name] = {"error": str(e)[:200]}
            print(f"[short:{name}] ERROR {e}", flush=True)
    needle = make_needle()
    print(f"[needle] prompt chars={len(needle)} (~{len(needle)} tokens)", flush=True)
    try:
        results["needle"] = call(needle, max_tokens=64)
        print(f"[needle] {results['needle']['content'][:80]!r}", flush=True)
    except Exception as e:
        results["needle"] = {"error": str(e)[:300]}
        print(f"[needle] ERROR {e}", flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
