# Stability battery — FP8 profiles (2026-10-12)

Same overnight battery, applied to the two **FP8 KV** profiles, on the live container
`llm-infer` (`:8080`) with image `llm-infer:hicache-b1259f3e` (SGLang v0.5.21, tree
`e00930c`; Rust radix core).

- **Mainline** `inference/kv-fp8-text-image.yml` — FP8 KV + vision, `mrr 2`.
- **Text-only** `inference/kv-fp8-text-only.yml` — FP8 KV, text-only, `mrr 4`.

## Headline — the prefill-CUDA-graph re-tune is unsafe on the FP8 mainline

The earlier per-profile prefill-graph re-tune (graph ON, `--cuda-graph-max-bs-prefill
2048`, `--mem-fraction-static 0.93`, ~0.78 GB runtime headroom) **crashed the mainline
during T0's `verify-hicache-thrash`**:

```
expandable_segments: memory mapping failed with OOM on device 0
torch.OutOfMemoryError: Tried to allocate 72.00 MiB ... 19.06 MiB free ... 31.31 GiB in use
Scheduler hit an exception  ->  kill_process_tree  ->  container restart
```

The ~1.1 GB prefill graph leaves too little headroom for the HiCache host↔device
reload. **Reverted the mainline to the proven `--disable-prefill-cuda-graph` +
`--mem-fraction-static 0.94` config**; T0 then passes cleanly (1.8 GB headroom).

The **text-only** profile (graph ON, `--cuda-graph-max-bs-prefill 3072`, frac 0.91,
~0.98 GB free) **survived** T0 + 4-way concurrency + a 4×200K sweep + soak — it has no
vision stack and a lower mem-fraction, so graph ON is safe there and was **kept**.
The **nvfp4** profile (graph ON, ~2.37 GB free) is unaffected.

## Results

| Tier | Test | Mainline (mrr 2) | Text-only (mrr 4) |
|------|------|------------------|-------------------|
| T0 | sched-latch | PASS | PASS |
| T0 | mamba-stash | PASS (2/2 safe) | PASS (2/2 safe) |
| T0 | hicache-thrash | PASS (after revert) | PASS |
| T1 | b2 | PASS | PASS |
| T1 | t1 = 4×48K | all-ok (bs4 N/A, mrr 2) | **PASS** (bs4, 1.73×) |
| T1 | t2 = mixed ~240K | all-ok (bs4 N/A) | **PASS** (retract 0) |
| T2 | RULER 64K/128K/156K/175K | hit | — |
| T2 | RULER multi @~234K Qwen | **5/5** | — |
| T3 | prefix cache | PASS (cached 2112) | — |
| T4 | mamba probe 1040 s | **PASS (0 garble)** | — |
| T5 | image ladder 1…96 | PASS; 128 over-context 400 | n/a (text-only) |
| T6 | quality vs FP8 baseline | on par | — |
| T8 | malformed / missing / abort×5 | PASS | — |
| T8 | 4-concurrent 4×120K/160K/200K | **all 200 (no crash)** | **all 200 (no crash)** |
| T9 | soak 600 s | PASS (103 req, 0 err) | PASS (109 req, 0 err) |

## The NVFP4 crash is NVFP4-KV-specific (confirmed)

The 4-concurrent long-prefill sweep that **crashes the nvfp4 profile** (a `RuntimeError`
in `radix_attention.py::_prepare_dequant_extend_workspace`, the NVFP4→FP8 prefill
dequant workspace) produces **HTTP 200 on both FP8 profiles**, even at 4×200K
(≈800K tokens, ≈3× the pool). FP8 KV is stored directly, so the NVFP4 dequant path is
never executed. See `../stability-20261012/REPORT.md` for the full analysis.

## Config deltas vs the committed v0.5.21 upgrade

- `inference/kv-fp8-text-image.yml`: prefill graph reverted to **off**
  (`--disable-prefill-cuda-graph`), `--mem-fraction-static` back to **0.94**.
- `inference/kv-fp8-text-only.yml`: unchanged (graph ON at 3072, fade 0.91) — validated.

## Artifacts (this directory)

```
t0-*/textonly-t0-*.txt          T0 gates (mainline after revert; text-only)
t1-concurrency-{b2,t1,t2}.txt   mainline T1
textonly-t1-{b2,t1,t2}.txt      text-only T1
tier2-ruler-*.txt / *.json      T2 RULER (mainline)
tier3-cache.txt tier6-quality.txt quality-fp8.json
tier5-image-ladder-full.txt     T5 image ladder
tier8-boundary.txt              T8 boundary
tier8c-sweep2.txt               T8 4-concurrent sweep (mainline, no crash)
textonly-sweep2.txt             T8 4-concurrent sweep (text-only, no crash)
tier1-concurrency-soak.txt      T9 soak (mainline)
textonly-soak.txt               T9 soak (text-only)
mamba-probe/ mamba-probe.log    T4 mamba probe (1040 s, 0 garble)
haystack.txt                    RULER haystack used
```

## Reproduction

```bash
# switch profile
docker compose -f inference/kv-fp8-text-image.yml up -d     # or ...-text-only.yml
# T0
cd inference/tools/acceptance
python3 verify-scheduling-patches.py --url http://localhost:8080/generate --container llm-infer
python3 verify-mamba-stash.py --expect safe --url http://localhost:8080/generate --container llm-infer
python3 verify-hicache-thrash.py --url http://localhost:8080/generate --container llm-infer
# T1 / T9
cd ../../tools
SGLANG_BASE=http://localhost:8080 C4_CONTAINER=llm-infer MODEL_NAME=qwen3.8-27b python3 concurrency-load.py t1
SGLANG_BASE=http://localhost:8080 C4_CONTAINER=llm-infer MODEL_NAME=qwen3.8-27b python3 concurrency-load.py soak
```
