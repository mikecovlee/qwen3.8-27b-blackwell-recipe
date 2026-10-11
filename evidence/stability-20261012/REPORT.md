# Stability battery — NVFP4 profile (2026-10-12)

Stress/soak results for the **`kv-nvfp4-text-image.yml`** profile (the "4 streams on
32 GB" variant), run overnight on the live container `llm-infer` (`:8080`).

## Scope & environment

- Profile: `inference/kv-nvfp4-text-image.yml`.
- Image: `llm-infer:hicache-b1259f3e` = SGLang **v0.5.21** base (digest `b1259f3e`,
  tree `e00930c`) + sched-latch + P3 patches baked in; Rust radix core.
- GPU: RTX PRO 4500 (Blackwell, SM120), 32 GB.
- Key server config: `--kv-cache-dtype nvfp4`;
  `--prefill-kv-cache-dequant-dtype fp8_e4m3` (SM120 resolves to prefill
  `flashinfer` + decode `trtllm_mha`); `--max-running-requests 4`;
  `--max-total-tokens 262144`; `--chunked-prefill-size 4096`;
  prefill CUDA graph **on** (`breakable`, P=4096, 1.53 GB, 2.38 GB free);
  `--schedule-policy hrrn`; mamba pool 16 / `extra_buffer_lazy` /
  `mamba-max-states-per-path 2`; HiCache ratio 2 / `write_through` / `kernel` /
  `page_first`.
- All probes are deterministic (temperature 0) and come from `tools/` and
  `inference/tools/acceptance/`.

## Results

| Tier | Test | Result |
|------|------|--------|
| T0 | sched-latch (`verify-scheduling-patches.py`) | **PASS** |
| T0 | mamba-stash (`verify-mamba-stash.py --expect safe`) | **PASS** (2/2 safe) |
| T0 | hicache-thrash (`verify-hicache-thrash.py`) | **PASS** (load_back +135296; D 0.4 s) |
| T1 | concurrency b2 | **PASS** (bs2 graph decode, eager 0) |
| T1 | concurrency t1 = 4×48K | **PASS** (bs4/bs2 = 1.80×) |
| T1 | concurrency t2 = mixed 2×80K+2×40K (~240K) | **PASS** (retract 0, assert 0) |
| T2 | RULER niah single 32K / 64K / 128K | **PASS** (hit) |
| T2 | RULER niah single near-limit 160K / 180K tok-meta (~208K/234K Qwen) | **PASS** (hit) |
| T2 | RULER multi-key/value @~234K Qwen | **PASS** (5/5 tasks score 1.0) |
| T2 | over-context (> 262144 tokens) | **PASS** (graceful HTTP 400) |
| T3 | prefix cache (`test-cache-report.py`) | **PASS** (cached 2112) |
| T4 | mamba saturation probe, 740 s | **PASS** (saturated avail=0, **0 garble**) |
| T5 | image ladder N = 1,4,8,…,96 | **PASS** (all ok; peak 31380 MiB) |
| T5 | image ladder N = 128 | graceful **HTTP 400** (over-context) |
| T6 | quality spotcheck vs FP8 baseline | PASS except known NVFP4 `math_mul` garble |
| T8 | malformed body / missing prompt / abort ×5 | **PASS** (clean 4xx, clean aborts) |
| T8 | 4-concurrent long prefill (≥ ~140K tok each) | **FAIL — server crash** (below) |
| T9 | soak 600 s (2–4 conc, mixed ctx) | **PASS** (109 req, 0 err/retract/assert/fatal) |

The 2026-09-11 mamba-saturation garble incident did **not** reproduce: the probe held
the mamba pool saturated (`avail=0.0`, queue 6–9) for 740 s and all three victim
sessions (V1/V1b/V2, 38 turns total) produced **zero** garbled responses.

## Headline finding — 4-concurrent long-prefill crash (NVFP4)

**Trigger.** Four concurrent requests each longer than ~140K tokens (combined
≥ ~560K ≈ **2.1×** the 262144-token context). Boundary sweep (4 concurrent):

| per-request | total | prefill graph ON | prefill graph OFF |
|---|---|---|---|
| 48K / 56K / 64K / 72K / 80K | ≤ 320K | OK | — |
| 120K | 480K | **OK** | **CRASH** |
| 160K | 640K | **CRASH** | **CRASH** |
| 200K / 240K | 800K / 960K | **CRASH** | **CRASH** |

**Symptom.** All four requests return HTTP 500; the scheduler raises, the process tree
is killed (`Scheduler hit an exception` → `kill_process_tree`), the container
auto-restarts (~70 s) and returns to health 200.

**Root cause (verbatim).**

```
RuntimeError: The expanded size of the tensor (X) must match the existing size (Y)
  at non-singleton dimension 0. Target sizes: [X, 4, 256]. Tensor sizes: [Y, 4, 256]
  at sglang/srt/layers/attention/radix_attention.py
     _prepare_dequant_extend_workspace -> pool.get_flashinfer_dequant_workspace_kv_buffer()
     dq_k[cur_token_idx_dq:cur_token_idx_dq + prev_len] = k_prev_fp8
```

`X` vs `Y`: 4×240K → 22080 vs 237568; 4×160K → 102080 vs 156864;
4×200K → 62080 vs 196608.

**Interpretation.** The NVFP4→FP8 prefill-KV-dequant workspace is allocated for one
token count but indexed with `prev_len` derived from another; under several
long-context concurrent prefills the two diverge. This is an **upstream SGLang bug in
the NVFP4 prefill-dequant path** — it is **not** caused by our sched-latch / P3
patches and **not** by the prefill CUDA graph (disabling the graph makes the crash
*worse*: it fires already at 4×120K). It is **NVFP4-KV-specific**; the FP8-KV
profiles never execute this dequant path.

**Safe zone (empirical).** 4-concurrent with combined prompt length ≤ ~480K tokens
(≤ ~120K each) — all pass, including the FCFS-free `t1`/`t2`/`soak` suites.

## Recommendations

- Keep the NVFP4 profile for its intended 4-stream use, but **avoid 4 concurrent
  prompts whose combined length approaches or exceeds ~2× the context window**
  (i.e. each ≳140K tokens). For 4-way *very-long-context* concurrency, prefer the
  FP8 profiles (`kv-fp8-text-image` / `kv-fp8-text-only`).
- File an upstream issue with the traceback above.
- No config change is required for normal workloads; the failing case is an extreme
  edge (four simultaneous half-context-or-larger prompts).

## Artifact index (this directory)

```
t0-t1-scheduling.txt            T0 sched-latch gate
t0-t3-mamba-stash.txt           T0 mamba-stash gate
t0-t4-hicache-thrash.txt        T0 hicache-thrash gate
t1-concurrency-{b2,t1,t2}.txt   T1 concurrency suites
tier1-concurrency-soak.txt      T9 soak (600 s)
tier2-ruler-*.txt / *.json      T2 RULER (single / multi / fwe, base + near-limit)
tier5-image-ladder-full.txt     T5 image ladder (1..96 ok, 128 over-context)
tier6-quality.txt               T6 quality spotcheck
quality-nvfp4.json              T6 raw quality results
tier8-boundary.txt              T8 malformed / missing / abort / unknown-model
tier8c-sweep2.txt               T8 4-concurrent threshold sweep (graph ON)
tier8d-nograph-sweep2.txt       T8 4-concurrent threshold sweep (graph OFF, isolation)
mamba-probe/  mamba-probe.log   T4 saturation probe (740 s, 0 garble)
haystack.txt                    RULER haystack used
```

## Reproduction

```bash
# T0 gates (test tweeps need the container name)
cd inference/tools/acceptance
python3 verify-scheduling-patches.py --url http://localhost:8080/generate --container llm-infer
python3 verify-mamba-stash.py --expect safe --url http://localhost:8080/generate --container llm-infer
python3 verify-hicache-thrash.py --url http://localhost:8080/generate --container llm-infer

# T1 / T9
cd ../../tools
SGLANG_BASE=http://localhost:8080 C4_CONTAINER=llm-infer MODEL_NAME=qwen3.8-27b python3 concurrency-load.py t1
SGLANG_BASE=http://localhost:8080 C4_CONTAINER=llm-infer MODEL_NAME=qwen3.8-27b python3 concurrency-load.py soak

# T2 RULER (near-limit targets are tiktoken lengths; Qwen tokens run ~1.3x)
RULER_HAYSTACK=evidence/stability-20261012/haystack.txt python3 ruler-niah.py 1 out.json 32768,131072,180000

# T4 mamba saturation (long)
SGLANG_BASE=http://localhost:8080 C4_CONTAINER=llm-infer python3 mamba-saturation-probe.py --exposure 600

# T8 4-concurrent crash threshold (see /tmp/opencode/sweep2.py logic)
```
