# Single-GPU LLM serving stack (SGLang + New API)

[中文说明](README-zh.md)

A reproducible, self-hosted LLM stack for **one NVIDIA Blackwell (sm120) GPU, 24 GB and
up (32 GB recommended)**: an OpenAI-compatible SGLang server running **Qwen3.8-27B**
with a 262144-token context, two concurrent streams by default (four with the NVFP4
profile), vision, and a New API gateway in front for multi-user keys, token billing
and failover.

Everything here is the distilled result of a long tuning process. The comments in the
compose files are deliberately short; the *why* lives in [Tuning](#tuning) and
[Known issues](#known-issues).

## Features

- **Full 262144-token context** resident on a single 32 GB card.
- Concurrent streams with near-linear batch scaling and zero retractions: 2 by
  default (FP8 KV profile), 4 with the NVFP4 profile.
- **FP8 KV cache + vision** as the 32 GB mainline default (full 262144-token context,
  mrr 2, prefill CUDA graph off, mixed chunk on; v0.5.21 tree, Rust radix core); **NVFP4 KV +
  vision** (*experimental*) for 4 streams on 32 GB (v0.5.21 tree; known occasional
  garbled-reasoning instability and a concurrent-long-prefill crash — see
  [NVFP4 caveats](#nvfp4-caveats)); **FP8 KV text-only** for high-concurrency (4 streams)
  text-only serving (v0.5.21 tree).
- **Hierarchical KV cache** (host-RAM L2) with fast re-admission after eviction.
- **Gateway**: per-user keys, 3-tier cache-aware pricing, automatic failover to a
  secondary upstream.
- **Reproducible**: one-click online install, plus an offline bundle workflow.

## Requirements

| | |
| --- | --- |
| GPU | 1x NVIDIA Blackwell (sm120), **32 GB recommended** (24 GB only with the NVFP4 profile) — see [GPU compatibility](#gpu-compatibility) |
| Driver | Recent NVIDIA driver + CUDA; `nvidia-container-toolkit` |
| Host RAM | 32 GB minimum, 64 GB recommended (host KV pool is ~17 GB) |
| Disk | ~90 GB (SGLang image ~48 GB + model ~22 GB + gateway) |
| OS | Modern Linux (developed on Ubuntu) |
| Software | Docker Engine, Docker Compose v2, Python 3 (tools only) |
| Model | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) (Apache-2.0) |

### GPU compatibility

The stack was built and measured on an RTX PRO 4500 (32 GB). Other Blackwell (sm120)
cards run the same images; pick a config variant by VRAM:

| Card | VRAM | Recommended variant |
| --- | --- | --- |
| RTX 5090D V2 / RTX PRO 4000 Blackwell | 24 GB | `kv-nvfp4-text-image` (only meaningful variant; expect to lower `--max-total-tokens`) |
| RTX 5090 | 32 GB | `kv-fp8-text-image` (default); `kv-nvfp4-text-image` if you want 4 streams |
| RTX PRO 4500 Blackwell (this repo) | 32 GB | `kv-fp8-text-image` (default, validated) |
| RTX PRO 5000 Blackwell | 48 GB | `kv-fp8-text-image`, optionally raise mrr to 4 (see [Tuning](#tuning)) |
| RTX PRO 6000 Blackwell | 96 GB | `kv-fp8-text-image`, and raise concurrency (see [Tuning](#tuning)) |

Why FP8 KV is the 32 GB default now: the KV pool budget is
`free − fraction slack − mm reservation − mamba pool`, and CUDA graphs are **not** in it
(captured afterwards, out of the slack). Keeping the prefill CUDA graph **off** on the
mainline — the ~1.1 GB graph no longer fits alongside vision + the full 262144 pool
(enabling it OOM'd the HiCache host↔device reload, observed 2026-10-12) — and rebalancing
to mrr 2 makes FP8 KV + vision + the full
262144 context fit in 32 GB with ~1.8 GB headroom — zero FP4 KV caveats, ~2x decode
bandwidth per stream. The mamba pool is 10 on the mainline (upstream sizing = ratio
5 × mrr, see [Mamba slot accounting](#mamba-slot-accounting-why-small-offline-requests-saturate-the-card-too));
the KV pool is capped by `--max-total-tokens`, not by memory, so the +2 slots cost zero
context. `kv-nvfp4-text-image` remains for those who need 4 streams on 32 GB (KV 8→5 GB,
FP4 caveats apply); if you never send images and want 4 streams, `kv-fp8-text-only` is
the long-running text-only high-concurrency profile. All three run on the v0.5.21 tree.

## Architecture

```
client (opencode / curl / any OpenAI SDK)
        │  Authorization: Bearer sk-...
        ▼
New API gateway  :8088          keys, quota, 3-tier billing, failover
        │  http://host.docker.internal:8080
        ▼
SGLang server    :8080          qwen3.8-27b, 262144 ctx, 2-4 streams
        │
        ▼
NVIDIA GPU (sm120, 32 GB)       NVFP4 weights + NVFP4/FP8 KV cache
```

The gateway is optional. SGLang alone speaks the OpenAI API at `:8080`.

## Quick start

### Online (machine has internet)

```bash
git clone https://github.com/mikecovlee/qwen3.8-27b-blackwell-recipe.git
cd qwen3.8-27b-blackwell-recipe
make online          # default VARIANT=fp8v (FP8 KV + vision, mrr 2, 32 GB)
# other profiles:     make online VARIANT=nvfp4   (4 streams on 32 GB, FP4 KV)
#                     make online VARIANT=fp8     (32 GB, text only, 4 streams)
```

`make online` runs `scripts/online/setup.sh`, which:
1. checks docker / GPU / compose,
2. prepares the SGLang image per variant — `fp8v` (default): pulls the pinned v0.5.21
   base by digest and **builds the patched derived image** `llm-infer:hicache-b1259f3e`
   from `inference/patches/` (scheduler false-latch + HiCache hybrid-Mamba
   patches); legacy variants: pull the old pinned image
   (digest, with a tag fallback) — and pulls the gateway image,
3. downloads the model to `$MODELS_DIR`,
4. generates `.env` (random `SESSION_SECRET`, `SGLANG_IMAGE` matched to the variant),
5. starts both services and waits for `/health`.

Then open the gateway at `http://localhost:8088`, create a user and a token, and point
your client at it — see [`gateway/opencode-config.md`](gateway/opencode-config.md).

<details>
<summary>Manual online install</summary>

```bash
cp .env.example .env          # edit MODELS_DIR and SESSION_SECRET
pip install -U "huggingface_hub[cli]"
hf download nvidia/Qwen3.8-27B-NVFP4 --local-dir "$MODELS_DIR/Qwen3.8-27B-NVFP4"
# mainline (fp8v): pinned v0.5.21 base + baked scheduler + HiCache patches
docker pull lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469
docker build -f inference/patches/hicache-mamba-fix/img-b1259f3e/Dockerfile \
  -t llm-infer:hicache-b1259f3e inference/patches
# legacy variants (nvfp4 / fp8 text-only) instead pull the old pinned tree:
#   docker pull lmsysorg/sglang@sha256:b91d664a8e4825afc16ab831c6035a6c88ac20ef8bd26da4fe2b9813a9f44376
docker pull calciumion/new-api:v1.0.0-rc.40
docker compose --env-file .env -f inference/kv-fp8-text-image.yml up -d
docker compose --env-file .env -f gateway/docker-compose.yml up -d
```
</details>

### Offline (air-gapped machine)

On a **networked** machine:

```bash
make export          # builds bundle/ : both images (fp8v: the patched derived image),
                     # the model, checksums (~70 GB); VARIANT= like online
```

Copy the whole `bundle/` directory and this repository to the offline host, then:

```bash
make offline BUNDLE=/path/to/bundle
# or: scripts/offline/setup.sh /path/to/bundle
```

`setup.sh` verifies checksums, `docker load`s the images, extracts the model, starts both
services and waits for health.

## Configuration variants

Three self-contained compose files live in `inference/`. They share every argument
except the ones below (`make check` enforces this):

| Argument | `kv-fp8-text-image.yml` (**mainline default**, 32 GB, v0.5.21 tree) | `kv-nvfp4-text-image.yml` (*experimental*, 32 GB, 4 streams) | `kv-fp8-text-only.yml` (32 GB, 4 streams) |
| --- | --- | --- | --- |
| image (default) | `llm-infer:hicache-b1259f3e` (derived; scheduler + HiCache patches baked in) | `llm-infer:hicache-b1259f3e` (same, baked) | `llm-infer:hicache-b1259f3e` (same, baked) |
| `--kv-cache-dtype` | `fp8_e4m3` | `nvfp4` | `fp8_e4m3` |
| attention backend | `--attention-backend flashinfer` | `--prefill-kv-cache-dequant-dtype fp8_e4m3` (SM120: prefill flashinfer + decode trtllm_mha) | `--attention-backend flashinfer` |
| vision | enabled | enabled | disabled (`language_model_only`) |
| image flags | `--mm-process-config`, `--image-processor-backend pil`, guard `image:128` | same | none |
| mrr / mamba pool / graph bs | **2 / 10 / 2** | 4 / 16 / 4 | 4 / 16 / 4 |
| prefill CUDA graph | off (`--disable-prefill-cuda-graph`) | on, capped at 4096 (=cps) | on, `--cuda-graph-max-bs-prefill 3072` |
| mixed chunk | **on** (`--enable-mixed-chunk`) | off | off |
| `--chunked-prefill-size` | `6144` (v0.5.21: 8192 OOMs the pool, see Tuning) | default (`4096`) | default (`4096`) |
| `--mamba-radix-cache-strategy` | `extra_buffer` | `extra_buffer_lazy` | `extra_buffer_lazy` |
| `--schedule-policy` | `hrrn` (upstream aging; waitfix retired 09-19) | `hrrn` | `hrrn` |
| `--mem-fraction-static` | `0.94` | `0.90` | `0.91` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | `expandable_segments:True` | `expandable_segments:True` |
| free VRAM after load (32 GB) | ~1.8 GB | ~2.37 GB | ~0.97 GB |

**Which one?** Start with **FP8 KV + vision** (mainline default): full 262144 context +
vision on 32 GB with zero FP4 KV caveats, at 2 concurrent streams (~2x decode bandwidth
each), on the v0.5.21 tree with baked-in scheduler + HiCache patches. The
**NVFP4 KV + vision** profile is **experimental**: pick it only when you need 4 streams
on the same card — or on a **24 GB** card, where its smaller KV footprint is the only
meaningful option — and accept both its **occasional garbled-reasoning instability** and
its **concurrent-long-prefill crash** (see [NVFP4 caveats](#nvfp4-caveats); keep the
concurrent matched-prefix sum under ~262K tokens). For 4-way *very-long-context*, use the
**FP8** profiles. Pick **FP8 text-only** if you never send images and want 4 streams. On
48 GB+, raise the mainline profile to mrr 4 / pool 20 / graph 4 (pool = 5 × mrr, see
[Tuning](#tuning)) and raise the prefill cap accordingly, then
re-run `tools/concurrency-load.py soak`.

```bash
make up                   # default: FP8 KV + vision, mrr 2 (32 GB)
make up VARIANT=nvfp4     # 32 GB, FP4 KV, 4 streams
make up VARIANT=fp8       # 32 GB, text only, 4 streams
```

The installers honor the same switch (`make online|offline VARIANT=...`); `fp8v` now
requires only 32 GB at preflight.

## Tuning

The interesting part of this repository is *why* the server is configured the way it is.
All of it was measured on the target card.

### Concurrency is a three-knob problem

`--max-running-requests` (mrr), `--max-mamba-cache-size` and `--cuda-graph-max-bs-decode`
must move together:

- The mamba state pool caps concurrency. Mainline (v0.5.21 tree, `extra_buffer` +
  overlap schedule): upstream auto-sizes the pool at **5 × mrr** (ratio 5 = 3 base + 2
  for overlap), i.e. pool 10 -> mrr 2, pool 20 -> mrr 4. The legacy profiles (old
  pinned tree, `extra_buffer_lazy`) budget 4 slots per request (`pool // 4`): pool 16 ->
  mrr 4.
- The CUDA-graph decode batch size must cover mrr, or larger batches silently fall back
  to eager mode (~60% slower).
- Default profiles: `kv-fp8-text-image` (mainline) runs `262144 / mrr 2 / pool 10 /
  graph 2 / extra_buffer` (prefill graph on, capped at 2048 tokens); the NVFP4 and
  text-only profiles run `262144 / 4 / 16 / 4 / extra_buffer_lazy` (prefill graph on,
  capped at 4096 / 3072 tokens) on the same v0.5.21 tree.
- On a 96 GB card you can raise all three together (mainline ratio: mrr 6 / pool 30 /
  graph 8, keeping graph ≥ mrr). The latch fix is validated at mrr 2 in production
  (v0.5.19 line, 24 h+; re-verified on v0.5.21 by the T1 gate) and at 4 streams on the legacy tree; re-run
  `tools/concurrency-load.py` (`t1`, `soak`) before trusting higher values.

### Mamba slot accounting (why small offline requests saturate the card too)

State slots are charged **per request and per radix path, not per token**: a live request
holds 1 active + up to 2 checkpoints on its path (track interval 256, capped by
`--mamba-max-states-per-path 2`) = 2-3 slots. On the legacy tree admission budgets 4
per request (measured: two streams in flight on pool 8 => used 5 / evictable 2 /
available 1).

On the mainline (v0.5.19→v0.5.21) the unified radix cache adds one more consumer: the **first
stash of a chunked-prefill request donates an extra slot** to the tree — peak per
request = own + locked + 1 donated (upstream sizing formula, pinned by its unit test
`test_mamba_donated_alloc_ratio.py`). Under `extra_buffer_lazy` the allocator budgets
only 2 slots per request, so a tight pool hits `assert slot is not None` ("Can not alloc
mamba cache") and takes the scheduler down — observed once in production (2026-09-15:
two streams holding 6/8 slots, a 150K chunked prefill admitted on the 2 freed slots,
its stash needing a 9th). The **E10 fix** (`extra_buffer` budgets 3, admission rejects
and queues instead of over-committing) + pool **10** = upstream ratio 5 × mrr 2. Measured
on the v0.5.19/20 line: dual-stream peak uses 8/10, leaving exactly the 2-slot stash
headroom. v0.5.21 reworks this accounting and auto-sizes the host mamba pool; re-verified
on v0.5.21 (T3 gate). Regression gate:
`inference/tools/acceptance/verify-mamba-stash.py` (original:
`evidence/mamba-stash-T3-0919/`).

Consequence (unchanged): an offline "reward-model scoring" loop - 60-1300-token prompts,
generation to a 700-token cap, ~30-50 req/min - needs 10-15 streams by Little's law and
pins any interactive server at full mrr with permanent eviction/reload churn on path
states. Short requests are cheap on KV and expensive on state.
Operational rule: run offline/batch workloads under their **own gateway token** with
**client-side concurrency <= 2** (or off-peak).

### The scheduler false-latch patch

Both pinned SGLang trees have the same false-latch bug (mainline v0.5.21
`b1259f3e` at `scheduler.py:3976`; earlier mainline anchors: 3887 on v0.5.20 `06e4f2ed`,
3661 on v0.5.19; the legacy `b91d664a` anchor is 3355 — older
anchors live in git history): a chunked-prefill
continuation (which already holds a request row and does not allocate a new one) is
counted in `can_run`, so it is compared against a budget derived from free rows. This
double-counts and sets `batch_is_full` early, capping effective concurrency at
`mrr - 1`.

The patch lives in `inference/patches/sched-latch-fix/`, one build directory per
pinned image (see its README for the re-anchor procedure). The mainline profile pins
the **derived image `llm-infer:hicache-b1259f3e`** (all-in-one build from
`inference/patches/`, also carrying the HiCache hybrid-Mamba patches below): the
scheduler patch is baked in via a `.pth` import — the base image ships a system
`sitecustomize.py` that silently shadows the old `PYTHONPATH=/patches` mount trick —
and it **self-verifies its anchor at startup and fails loudly on drift**. The legacy
profiles still pin `b91d664a` and load `img-b91d664a/sitecustomize.py` from the
`/patches` mount (matches on line number + function name — re-check the anchor after
any image upgrade or the hook silently no-ops; safe: it just falls back to `mrr - 1`).

The suppression rule: only suppress the false latch when the real number of new
admits in a pass is below the free rows at the start of the pass — so it never
over-admits and is safe at any `mrr`. The bug needs an in-flight chunked prefill
(prompt longer than `chunked_prefill_size`: 2048 legacy / 6144 mainline) plus a
second request arriving mid-prefill; prompts at or below the chunk size never chunk
and are unaffected.

### Context pool budget and the prefill CUDA graph

The pool is profiled as
`pool_bytes = free_after_weights - mem_fraction slack - mm reservation - mamba pool`,
and **CUDA graph capture is not in that formula** - graphs are captured afterwards and
spend the `slack`. So while the pool is pinned by `--max-total-tokens`,
`--mem-fraction-static` moves slack around but not the pool; but when the *profiled*
value is the constraint (as with the graph on), fraction and graph cost move the pool
directly. On v0.5.21 the prefill CUDA graph is **on in the NVFP4 and text-only profiles**
(shape-capped via `--cuda-graph-max-bs-prefill`: NVFP4 4096, text-only 3072) but **off on
the mainline**: with vision + the full 262144 pool at `--mem-fraction-static 0.94`, the
~1.1 GB graph leaves too little slack and enabling it OOM'd the HiCache host↔device reload
(observed 2026-10-12), so the mainline keeps `--disable-prefill-cuda-graph`. Prefill chunks
larger than a cap replay eagerly (measured 2026-09-11: chunked prefill is compute-bound, not
launch-bound). Measured headroom after capture: mainline `0.94`, graph off, ~1.8 GB free;
text-only `0.91` -> graph 1.34 GB, ~0.98 GB free; NVFP4 `0.90` -> graph 1.53 GB, ~2.37 GB free. The pool and cold-prefill
headroom remain a 1:1 trade (table below measured with NVFP4 KV / mrr 4, same shape):

| `max-total-tokens` | free VRAM | cold prefill | 4 concurrent |
| --- | --- | --- | --- |
| **262144 (this stack)** | 2.65 GB | ~252K | 4 x 65K |
| 300000 | ~1.9 GB | ~180K | 4 x 75K |
| 320000 | 1.54 GB | ~140K | 4 x 80K |

### Mixed chunk: decode rides the prefill batch

The mainline enables `--enable-mixed-chunk`. Without it a chunked prefill owns the GPU and
every concurrent decode request freezes for the whole prefill. Measured on v0.5.21 (mrr 2,
cps 6144, a ~40K-token cold prefill A injected 2 s into B's 512-token stream):

| | mixed chunk off | on |
| --- | --- | --- |
| B's worst inter-token gap | **9.5 s** (one freeze) | **2.3 s** (6 gaps) |
| B's total stall | ~9.5 s | ~9.5 s |
| B's wall / ITL p50 | 22.0 s / 23.3 ms | 22.0 s / 23.4 ms |
| A's prefill wall | 9.9-10.0 s | 9.9-10.1 s |

The win is **streaming smoothness, not throughput**: mixed chunk carries one decode token
per running request per prefill step, and a 6144-token step takes ~1.5-2.3 s, so the same
total stall is split into per-chunk hiccups instead of one 9.5 s hang. Wall clock cannot
improve because 512 decode tokens need 512 steps. It costs A nothing. Validated by T1/T3/T4,
`concurrency-load b2/t1/t2`, a 600 s soak (103 requests, 0 errors, 0 restarts) and a
functional-correctness check — batch composition changes kernel numerics, so outputs are not
bit-identical, but the answers stay correct. The sched-latch patch is unaffected: the flag
only trims the prefill token budget (`rem_input_tokens` / `rem_chunk_tokens` /
`create_prefill_budget`), leaving the row-and-quota accounting the latch guard reads
unchanged. Upstream forbids mixed chunk only together with speculative decoding
(`arg_groups/validation_hook.py`), which this stack does not use, and `Mamba2AttnBackend`
explicitly supports mixed prefill + decode.

### Hierarchical cache

`--enable-hierarchical-cache --hicache-ratio 2` keeps an L2 KV cache in host RAM. It
turns a 190K-token re-admission from ~109 s into <1 s with no decode/TTFT regression.
`ratio 2` uses ~17 GB of host RAM; raise it for a larger L2 at the cost of RAM.

[2026-09-17, v0.5.19] On this hybrid GDN (Mamba) model the host tier only actually served
requests after the `hicache-mamba-fix` patch build + `--chunked-prefill-size 6144` +
an explicit `SGLANG_HICACHE_MAMBA_SIZE_GB=7.0`: an evicted 36.6K session returned from host RAM
in **0.26 s** (8.76 s by full re-prefill before — 34x), branch re-admission dropped
25.1 s -> 3.09 s, and 2x68.5K concurrent prefills ran clean. **On v0.5.21 the explicit
knob is no longer needed** — see the next section.

### The HiCache hybrid-Mamba fix

Stock SGLang (v0.5.19) did not actually reuse the host tier on this hybrid GDN (Mamba)
model: chunked prefills were never backed up, the mamba anchor pool was sized below the
upstream criterion `kv_pool_tokens * hicache_ratio / chunked_prefill_size`, and a
starved mamba allocation could assert-crash the scheduler. The
`inference/patches/hicache-mamba-fix/` build carried three fixes for that tree (chunked
write-through backport #36647; skip-instead-of-assert on mamba exhaustion #36770; the
`SGLANG_HICACHE_MAMBA_SIZE_GB` host-pool knob), plus a fourth — honest host-hit
accounting — retired on v0.5.20 (`host_loaded_length` / `materialized_host_hit_len()`).

**On v0.5.21 most of this is absorbed upstream** (see that directory's README for the
retirement table):
- **P1 (chunked write-through backport #36647) — retired since v0.5.21.** v0.5.21 has a
  first-class chunked-prefill stash + backup pipeline
  (`scheduler.stash_chunked_request` → `maybe_cache_unfinished_req(chunked=True)` →
  `cache_unfinished_req(chunked=True)` + `flush_pending_backups()`).
- **C (`SGLANG_HICACHE_MAMBA_SIZE_GB` knob) — obsolete since v0.5.21.** v0.5.21 sizes the
  host mamba pool from the HiCache ratio / host budget (`hicache_auto_size.py`); on this
  box the auto pool (~20 host + 10 device slots) already passes the T4 gate, so the
  explicit knob is dropped from the profile. The C code stays baked in the image but
  dormant (env unset) as a zero-cost fallback.
- **P3 (mamba exhaustion assert #36770) — still carried.** v0.5.21 still has the fatal
  `assert slot is not None` at `unified_cache/components/mamba.py:509`, so the
  skip-instead-of-assert backport is re-anchored to `img-b1259f3e`.

Mainline runs `--chunked-prefill-size 6144` (8192 OOMs the pool) with no explicit
mamba-GB env. The regression gate `verify-hicache-thrash.py` demands a real host load-back
(`sglang:load_back_tokens_total{pool="kv"}`), not just a fast answer; v0.5.21 passes
(original measurements in `evidence/hicache-mamba-fix-0917/`).

### NVFP4 caveats

> **Experimental.** The NVFP4 profiles are not the supported default — they exist only
> for 4 concurrent streams on 32 GB (or a 24 GB card). On SM120 they carry two known
> defects that have **no fix**: a *hard crash* under concurrent long prefills (first
> bullet) and *occasional garbled reasoning* (second bullet). Prefer the FP8 profiles
> unless you specifically need NVFP4.

- **Concurrent-long-prefill crash (upstream bug, open)**: four concurrent requests whose
  *matched-prefix sum* exceeds roughly one context crash the scheduler — all four return
  HTTP 500, the process tree is killed and the container restarts (~70 s). Empirical
  trigger: 4 concurrent prompts of ≳140K tokens each (combined ≳ ~2x the 262144 context);
  4x120K is safe, 4x160K/200K/240K crash. Root cause: the NVFP4->FP8 prefill dequant
  workspace is sized for a **single** context
  (`fp4_kv_cache_quant_method.create_buffers`: `m = max_total_tokens + page_size`) but
  `memory_pool._prepare_dequant_extend_workspace` indexes it **cumulatively across the
  batch** as `Σ(matched_prefix + chunk)`; once that exceeds the buffer the write is out
  of range (`RuntimeError: The expanded size ... [X,4,256] vs [Y,4,256]`; X+Y ≈ 262080 ≈
  `max_total_num_tokens`). No CLI flag avoids it — on SM120, nvfp4 KV + flashinfer
  prefill is unconditionally routed to this workspace (native NVFP4 prefill needs SM100) —
  and disabling the prefill CUDA graph makes it **worse** (crashes already at 4x120K).
  **Safe envelope: keep the concurrent requests' combined matched-prefix length below
  ~262K tokens (empirically, each of 4 concurrent prompts ≤ ~120K).** For 4-way
  *very-long-context*, use the FP8 profiles.
- **Occasional garbled reasoning under multi-session churn (open, observed in production)**: inside a 40-minute
  window where mrr 4 was saturated and the mamba pool pinned at `available = 0` (constant
  eviction / host reload of path states), one long agent session intermittently produced
  token-soup reasoning, once running away to 20000 tokens. 26 minutes of controlled
  synthetic saturation reproduced 0 events; the leading explanation is a rare FP4-KV
  sampling derailment amplified by the client feeding its own garbage back into history.
  Unproven either way - which is one more reason FP8 KV is the default profile. Harness:
  `tools/mamba-saturation-probe.py`.
- **KV host-pool sizing is 2x oversized for NVFP4**: the host pool is computed at
  32 KB/token, as if KV were not packed, so it uses the same RAM as FP8. This is an
  upstream sizing bug; patching it breaks the host->GPU reload kernel, so it is left as is.
- The checkpoint ships without calibrated KV scales, so NVFP4 uses a global scale of 1.0
  with per-block fallback. Measured quality is unaffected.
- **MTP / speculative decoding is not viable here**: the draft model needs ~5.5 GB and
  collapses the KV pool, and NVFP4 verify hits an unsupported path.

## Known issues

- **The NVFP4 KV profile crashes under concurrent long prefills** (upstream sizing bug in
  the NVFP4 prefill-dequant workspace; see [NVFP4 caveats](#nvfp4-caveats)). Four
  concurrent prompts of ≳140K tokens each (combined ≳ ~2x the context) return HTTP 500 and
  restart the container. Keep the concurrent matched-prefix sum under ~262K tokens, or use
  the FP8 profiles for 4-way long context.
- **The NVFP4 KV profile occasionally emits garbled reasoning** (intermittent, observed in
  production; see [NVFP4 caveats](#nvfp4-caveats)). Use it only for 4 streams
  on 32 GB, or on a 24 GB card where it is the only meaningful variant.
- **`page_size` becomes 64 with `trtllm_mha`**, after which the default
  `--mamba-max-states-per-path -1` lets mamba states accumulate per radix path, filling
  the 16-slot pool and deadlocking new allocations (container restart). All profiles set
  `--mamba-max-states-per-path 2` (a zero-VRAM behavioral cap that matches the
  extra-buffer ping-pong design; kept at 2 on the mainline `extra_buffer` too — note it
  bounds *tree* checkpoints only, not live request slots). Always re-run a soak test
  after changing the KV recipe.
- **v0.5.19 + `extra_buffer_lazy` + pool 8 could assert-crash the scheduler** ("Can not
  alloc mamba cache") when a chunked-prefill stash needed a donated slot from a tight
  pool with no evictable victim (observed once in production, 2026-09-15). Fixed by E10:
  the mainline runs `extra_buffer` (allocator budgets 3/request; admission rejects and
  queues instead of over-committing) + pool 10 (upstream ratio 5 × mrr 2). The assert
  itself is upstream fail-loud design (still present on main); the mainline patch build
  additionally ships the upstream skip-instead-of-assert backport (#36770, counted via
  `radix_cache_aux_alloc_failed_total`). Regression gate:
  `inference/tools/acceptance/verify-mamba-stash.py` (latest evidence:
  `evidence/mamba-stash-T3-0919/`; the 09-15 A/B runs are in git history).
- **`--mm-process-config` uses pixel *area*, not edge length.** `image.max_pixels` is
  ignored by this processor build; use `image.size.longest_edge` (2097152 = 2 Mpx area).
- **Images need `--image-processor-backend pil`.** The GPU image processor resizes all
  images in one pass as fp32 tensors and OOMs the tokenizer process; `pil` preprocesses
  on CPU. The count guard `--limit-mm-data-per-request '{"image":128,"video":0}'` exists
  because it fires **before** preprocessing while the context-length check fires
  **after**. 128 sits just past the natural ceiling: each 2 Mpx image costs ~2050 tokens,
  so ~127 images already exhaust the 262144 window, and VRAM never binds (GPU peak
  measured flat from 4 to 96 images).
- **New API routes by the `abilities` table**, not `channels`. Editing channels directly
  (priority / model list) has no effect unless abilities is updated too.
- **Firewall failover latency**: with ufw, an unlistened port is DROP (30 s SYN timeout),
  so "port blocked" failures take ~30 s to fail over; a dead process (RST) fails over in
  under a second.

## Benchmarks

Measured on the target card (RTX PRO 4500, 32 GB, sm120). Full results in
[`evidence/README.md`](evidence/README.md).

### NVFP4 vs FP8 (this deployment)

| Scenario | FP8 | NVFP4 |
| --- | --- | --- |
| single-stream decode @1K | 42.6 | 42.4 tok/s |
| single-stream decode @64K | 38.5 | 39.6 tok/s |
| cached TTFT | 0.05 s | 0.06 s |
| batch 4 @1K | 153.4 | 152.3 tok/s |
| batch 4 @48K | 118.6 | 128.2 tok/s |
| soak 600 s | 106 req / 0 err | 106 req / 0 err |

These figures exclude the concurrent-long-prefill case, where the NVFP4 profile crashes
and the FP8 profiles are unaffected — see [NVFP4 caveats](#nvfp4-caveats).

### FP8 KV + vision on 32 GB (pool-8 baseline, 2026-09-11)

| Check | Result |
| --- | --- |
| boot accounting | KV 8.0 GB + mamba 0.65 GB, ~1.91 GB headroom after graph capture |
| prefill graph off vs on (same stack) | 0 to +1% (1K/14K/70K; TTFT identical) |
| cold 166K prefill | 87.3 s (1905 tok/s), zero retractions |
| single-stream decode @1K | 41-43 tok/s (mrr 2) |
| mamba slots, 2 streams in flight | used 5 / evictable 2 / available 1 (pool 8) |
| images (2 Mpx each) | 1-96 pass, GPU peak flat; 128 rejected by the context window |
| quality spot check | all pass incl. 104K needle |

### E10 re-validation (mainline: v0.5.19, `extra_buffer`, pool 10, 2026-09-15)

| Check | Result |
| --- | --- |
| boot accounting | KV 8.0 GB / 262144 tokens **unchanged** (flag-capped) + mamba 0.80 GB; 2.43 GB free after pools, ~1.78 GB after graph capture |
| T3 stash-crash matrix (`verify-mamba-stash.py --expect safe`) | 153K chunked prefill: avail ≥ 2/10 throughout, zero forced evictions, zero restarts |
| single-stream decode | 43.1 tok/s @1K; 38.8 @64K (TTFT 18.7 s) |
| dual-stream decode (engine-side, bs-2 graph) | 69.7 tok/s median, zero retractions |
| cold 166K prefill | 87.1 s (vs 87.3 s at pool 8) |
| cache-hit billing (second shot) | 99.4% cached, TTFT 0.13 s |
| 150K × 2 contention | zero retractions/asserts (the second 150K queues by design: 300K > 262144 pool) |
| T1/T2 scheduling patches | PASS (latch interleave + wait-boost ordering) |
| 9.5 h production soak | mamba peak 8/10 at dual stream, zero asserts, zero restarts |

### Throughput vs context

| Context | 2K | 4K | 64K | 128K | 240K |
| --- | --- | --- | --- | --- | --- |
| prefill (tok/s) | 5716 | 5737 | 3300 | 2203 | 1386 |
| decode (tok/s) | 42.0 | 41.8 | 39.3 | 37.0 | 33.2 |
| TTFT (s) | 0.38 | 0.74 | 19.2 | 57.5 | 171.3 |

Decode at 2K/4K is measured over 16 generated tokens; longer contexts use 128-256.

### Quality

- **GSM8K**: quantizer reports **97.27%** (1283/1319) on 4x GB300; this single-GPU
  deployment measured **97.5%** (195/200) with the same thinking protocol.
- **Terminal-Bench 2.1**: quantizer 73.81%, upstream Qwen card 73.0.
- **RULER long context**: single-needle NIAH 100% at 32K/64K/128K/**234K**; multi-needle
  NIAH and the harder `freq_words_extraction` also 100% at 234K — effective context
  length is at least 234K within the 262144 window.
- Quality spot check: 13/13, including a 104K needle.

Conclusion: on accuracy benchmarks, NVFP4 weights + NVFP4 KV introduce no measurable
regression (this is separate from the NVFP4 stability issue above).

## Gateway (New API)

See [`gateway/`](gateway/). Highlights:

- **Pricing**: 3 tiers derived from the upstream price — input, cached input, output.
  `quota = round((uncached_input + cached_input * CacheRatio + output * CompletionRatio) * ModelRatio)`.
  This depends on SGLang's `--enable-cache-report` (otherwise hits are billed at full price).
- **Failover**: channel 1 = SGLang (`priority 10`), channel 2 = a secondary upstream
  (`priority 0`). Set `RetryTimes=2`; the default `0` does **not** retry. Routing is driven
  by the `abilities` table.
- **Backups**: `docker cp llm-gateway:/data/one-api.db ./backup.db` (copy `-wal` too).
  The SQLite DB holds plaintext token keys, so treat `data/` as a credential store.
- **Security before exposing to the internet**: put a TLS reverse proxy in front (the
  gateway is plain HTTP), and note the upstream SGLang has no API key by default.

## Tools

All tools read `SGLANG_BASE` and `MODEL_NAME` (plus `GW_BASE`/`GW_KEY` for the gateway
scenarios, and `C4_CONTAINER`/`LLM_SERVICE_ROOT` for the engine-stats scenarios).

| Tool | Purpose |
| --- | --- |
| `tools/benchmark.py` | prompt-eval / decode throughput vs context length |
| `tools/concurrency-load.py` | concurrency gate + soak driver (`b2\|t1\|t2\|t3\|soak`) |
| `tools/gsm8k.py` | GSM8K accuracy (needs a GSM8K test JSONL, see `GSM8K_DATA`) |
| `tools/ruler-niah.py` | RULER single-needle NIAH (completion protocol) |
| `tools/ruler-niah-multi.py` | RULER multi-needle / multikey / multivalue NIAH |
| `tools/ruler-fwe.py` | RULER `freq_words_extraction` (hard aggregation) |
| `tools/quality-spotcheck.py` | fixed deterministic prompts, FP8 vs FP4 diffable |
| `tools/test-cache-report.py` | verifies `cached_tokens` reporting (billing dependency) |
| `tools/image-limit-ladder.py` | image-count ladder with VRAM-peak sampling (finds the real ceiling) |
| `tools/mamba-saturation-probe.py` | synthetic RM-style saturation + garble detector (investigation harness) |

```bash
make bench     # throughput
make eval      # prints how to run the quality / long-context evaluations
make check     # compose consistency + secret scan
```

The tools need Python 3 and `requests`; the RULER tools additionally need a haystack text
file (`RULER_HAYSTACK`, default `haystack.txt`) and the packages `tiktoken` and `wonderwords`.

## Repository layout

```
.
├── README.md / README-zh.md
├── LICENSE                     Apache-2.0
├── Makefile  .env.example
├── inference/
│   ├── kv-fp8-text-image.yml       mainline default: FP8 KV + vision (32 GB, mrr 2,
│   │                               v0.5.21 tree, hrrn, Rust radix core, scheduler + hicache fix: cps 6144)
│   ├── kv-nvfp4-text-image.yml     legacy: NVFP4 KV + vision (32 GB, mrr 4, old pinned tree)
│   ├── kv-fp8-text-only.yml        legacy: FP8 KV, text only, 4 streams (old pinned tree)
│   ├── patches/sched-latch-fix/    scheduler false-latch patch (one build per pinned image)
│   ├── patches/hicache-mamba-fix/  HiCache patches (write-through, alloc degrade, host pool)
│   └── tools/acceptance/           acceptance gates (T1 latch, T3 mamba stash, T4 hicache)
├── gateway/
│   ├── docker-compose.yml
│   └── opencode-config.md / opencode-config-zh.md
├── tools/                      benchmark / eval / ops scripts
├── evidence/                   benchmark and eval results
└── scripts/
    ├── online/setup.sh
    ├── offline/export.sh  offline/setup.sh
    └── check.sh
```

## License

Apache-2.0 (see [LICENSE](LICENSE)).

This project configures and wraps third-party software; it does not redistribute it:
[SGLang](https://github.com/sgl-project/sglang) (Apache-2.0),
[New API](https://github.com/Calcium-Ion/new-api) (AGPLv3),
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and
[nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) (Apache-2.0).
If you expose the gateway to third parties, you are responsible for compliance
(licensing, content safety, logging, etc.).
