#!/usr/bin/env python3
"""sm120 cuBLASLt FP8 bmm 微探针：复现旧机崩溃路径的最小用例。

裁决问题：flashinfer.bmm_fp8(backend="cublas") 在本机 (RTX PRO 4500, GB203)
+ 镜像内 cublas 组合下能否工作。

用法（容器内）:
    docker run --rm --gpus all -v $PWD:/w IMAGE python3 /w/probe-bmm-fp8.py
退出码：0=探针通过(免补丁路线可行)  非0=崩溃/不支持(需 sitecustomize 钩子)
"""
import sys

import torch
from flashinfer import bmm_fp8

import flashinfer

dev = "cuda"
print(f"flashinfer={flashinfer.__version__} torch={torch.__version__} "
      f"cc={torch.cuda.get_device_capability(0)} device={torch.cuda.get_device_name(0)}")

# 真实 q_proj 形状: out=24*256=6144, in=5120；M 覆盖 decode(2)/prefill(512)
CASES = [(2, 5120, 6144), (512, 5120, 6144), (2, 5120, 1024), (2, 6144, 5120)]


def mk(m, k, n, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    a = torch.randn(m, k, device=dev, generator=g).clamp_(-2, 2).to(torch.float8_e4m3fn)
    b = torch.randn(k, n, device=dev, generator=g).clamp_(-2, 2).to(torch.float8_e4m3fn)
    return a.unsqueeze(0), b.unsqueeze(0)


fail = 0
for i, (m, k, n) in enumerate(CASES):
    A, B = mk(m, k, n, i)
    sa = torch.ones(1, device=dev, dtype=torch.float32)
    sb = torch.ones(1, device=dev, dtype=torch.float32)
    try:
        out = bmm_fp8(A, B, sa, sb, torch.bfloat16, backend="cublas")
        torch.cuda.synchronize()
    except Exception as e:
        print(f"FAIL cublas bmm_fp8 M={m} K={k} N={n}: {type(e).__name__}: {e}")
        fail += 1
        continue
    try:
        ref = bmm_fp8(A, B, sa, sb, torch.bfloat16, backend="cutlass")
        rel = (out.float() - ref.float()).norm() / (ref.float().norm() + 1e-6)
        print(f"OK   cublas bmm_fp8 M={m} K={k} N={n}  vs cutlass rel_err={rel:.4f}")
        if rel > 0.05:
            print(f"WARN 数值偏差过大 M={m} K={k} N={n}")
            fail += 1
    except Exception as e:
        print(f"OK   cublas bmm_fp8 M={m} K={k} N={n}  (cutlass 对照不可用: {e})")

sys.exit(1 if fail else 0)
