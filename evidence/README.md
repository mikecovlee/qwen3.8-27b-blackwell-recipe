# Evidence

[中文](README-zh.md)

Results backing the numbers in the top-level [README](../README.md). All measurements were
taken on the target card (single 32 GB Blackwell sm120, NVFP4 weights, 262144-token pool).

## Throughput vs context

`tools/benchmark.py`, thinking on, single stream.

| Context | 2K | 4K | 64K | 128K | 240K |
| --- | --- | --- | --- | --- | --- |
| prefill (tok/s) | 5716 | 5737 | 3300 | 2203 | 1386 |
| decode (tok/s) | 42.0 | 41.8 | 39.3 | 37.0 | 33.2 |
| TTFT (s) | 0.38 | 0.74 | 19.2 | 57.5 | 171.3 |

Prompt token counts: 2160 / 4233 / 63432 / 126778 / 237477. Decode at 2K/4K is measured
over 16 generated tokens; longer contexts use 128-256.

## Concurrency and stability (NVFP4, mrr 4)

`tools/concurrency-load.py`.

| Scenario | Result |
| --- | --- |
| 4 x 1K, 2048 gen | batch-4 decode median 152.3 tok/s, eager 0 |
| 2 x 48K | batch-2 median 72.6 tok/s, retract 0 |
| 4 x 48K | batch-4 median 128.2 tok/s, wall 114.9 s, retract 0 |
| 2 x 80K + 2 x 40K (~240K) | wall 143.2 s, retract/assert/fatal 0 |
| single stream 1K / 64K | 42.4 / 39.6 tok/s |
| cached TTFT | 0.06 s (cached 2176 of 2189) |
| soak 600 s | 106 requests, 0 errors, 0 restart |

Before `--mamba-max-states-per-path 2`, the soak crashed within 1-2 minutes
(`Can not alloc mamba cache`); see the top-level README "Known issues".

## Scheduler false-latch fix

Before the patch, `mrr=4` admitted only 3 real streams. After mounting
`inference/patches/sched-latch-fix`:

| Scenario | Result |
| --- | --- |
| 4 x 1K | all admitted, batch-4 median 153-158 tok/s |
| 6 x 1K (oversubscribed) | 3 running + 3 queued, no crash |
| 4 x 48K | wall 108-115 s, retract 0 |
| soak 600 s | 106 requests, 0 errors |

The hook logged `suppressed false latch` on the affected passes.

## GSM8K

`tools/gsm8k.py`, thinking on, temp 1.0 / top_p 0.95 / top_k 20, first 200 questions.

| Metric | Value |
| --- | --- |
| correct | 195 / 200 |
| accuracy | **97.5%** |
| wall | 718.7 s |
| quantizer reference (full 1319 questions) | 97.27% |

## Quality spot-check

`tools/quality-spotcheck.py`, deterministic (temperature 0).

| Case | Result |
| --- | --- |
| 12 short prompts (math, logic, translation, JSON, code, formatting) | 12/12 correct |
| long-context needle at ~57.6K prompt tokens | correct |

## RULER long context

Completion protocol (`tools/ruler-niah.py`, `ruler-niah-multi.py`, `ruler-fwe.py`).

### Single-needle NIAH (needle depth 90%)

| Length | Score |
| --- | --- |
| 32K | 2/2 |
| 64K | 2/2 |
| 128K | 2/2 |
| 240K | 2/2 |

### Multi-needle / multikey / multivalue @ 240K

| Task | Score |
| --- | --- |
| niah_single_1 | 100% |
| niah_single_2 | 100% |
| niah_single_3 | 100% |
| niah_multikey_1 | 100% |
| niah_multivalue | 100% |

### freq_words_extraction (hardest aggregation)

| Length | Score |
| --- | --- |
| 32K | 2/2 |
| 240K | 2/2 |

## NVFP4 vs FP8

Same deployment, only the KV cache dtype (and vision) changed.

| Scenario | FP8 | NVFP4 |
| --- | --- | --- |
| single-stream decode @1K | 42.6 | 42.4 tok/s |
| single-stream decode @64K | 38.5 | 39.6 tok/s |
| cached TTFT | 0.05 s | 0.06 s |
| batch 4 @1K | 153.4 | 152.3 tok/s |
| batch 4 @48K | 118.6 | 128.2 tok/s |
| soak 600 s | 106 req / 0 err | 106 req / 0 err |

## FP8 KV + vision on 32 GB (current default profile, 2026-09-11)

Boot accounting: KV 262144 @ 8.0 GB, mamba pool 8 @ 0.65 GB, ~1.91 GB headroom after
decode-graph capture. Prefill graph off vs on: 0 to +1% (1K/14K/70K; TTFT identical).

| Scenario | Result |
| --- | --- |
| cold 166K prefill | 87.3 s (1905 tok/s), retract 0 |
| single-stream decode @1K (mrr 2) | 41-43 tok/s |
| mamba slots, 2 streams in flight | used 5 / evictable 2 / available 1 (pool 8) |
| images 1..96 (2 Mpx each) | all pass, GPU peak flat (~31168/32623 MiB) |
| images x128 | HTTP 400 from the 262144-token context check (262453 tokens) |
| quality spot check incl. 104K needle | pass |

## Garbling investigation harness (NVFP4 era, 2026-09-11)

During a window where mrr 4 was saturated by an offline scoring loop (mamba pool pinned
at `available = 0`), one interactive session intermittently produced token-soup
reasoning (worst case a 20000-token runaway). Two controlled saturation rounds
(`tools/mamba-saturation-probe.py`: RM-style pressure workers + long/short multi-turn
victims with an online garble detector) held `usage 0.75 / available 0` for 26 minutes
with eviction/reload visibly exercised (victim `cached_tokens` collapsing between turns)
and produced **0 garble events** across 29 victim turns.

Outcome: slot saturation alone is not sufficient to reproduce; the
leading explanation is a rare FP4-KV sampling derailment self-amplified by the client
feeding garbage back into history. The default profile moved to FP8 KV, which removes
the entire class (and page 64 / trtllm_mha alongside it).

## Mamba stash crash & E10 fix (2026-09-15)

Production crash 13 h after the v0.5.19 cutover: `_alloc_mamba_slot` assert during
`stash_chunked_request` (pool 8; `extra_buffer_lazy` admission coefficient 2 < peak 3
slots/req — upstream FIXME plus a unit test pin this as fail-loud by design). Fix = E10:
`--mamba-radix-cache-strategy extra_buffer` + `--max-mamba-cache-size 10` (upstream
auto-sizing ratio 5 × mrr 2). KV pool unchanged at 262144 tokens (flag-bound; the
+0.16 GB is absorbed by static-budget headroom). Twin verification, regression numbers
and the new `verify-mamba-stash.py` (T3) upgrade gate:
[`mamba-stash-T3-0915/`](mamba-stash-T3-0915/).

## HiCache hybrid-Mamba fix & host-pool sizing (2026-09-17)

v0.5.19's HiCache did not actually reuse the host tier for this hybrid GDN (Mamba)
model: a 190K continuation still measured ~109 s, and the host-hit counters were
phantom (device-resident tokens were credited to the host tier). Chunked prefills were
ineligible for write-through backup, the mamba anchor pool was undersized by the
upstream criterion (`262144/cps` anchors needed), and a starved allocation could
assert-crash the scheduler. After the `hicache-mamba-fix` patch build + `cps 6144` +
`SGLANG_HICACHE_MAMBA_SIZE_GB=7.0`: an evicted 36.6K session returns from host RAM in
**0.26 s** (vs 8.76 s recompute — 34×), the branch case drops 25.1 s → 3.09 s, and
2×68.5K concurrent prefills run clean. The regression gate
(`verify-hicache-thrash.py`) now demands a real host load-back
(`sglang:load_back_tokens_total{pool="kv"}`), not just a fast answer. Evidence:
[`hicache-mamba-fix-0917/`](hicache-mamba-fix-0917/).
