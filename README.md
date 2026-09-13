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
- **FP8 KV cache + vision** as the 32 GB default (full 262144-token context, mrr 2,
  prefill CUDA graph off); **NVFP4 KV + vision** for 4 streams on 32 GB (known
  occasional garbled-reasoning instability); **FP8 KV text-only** for high-concurrency
  (4 streams) text-only serving.
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
(captured afterwards, out of the slack). Disabling the prefill CUDA graph (1.19 GB,
measured 0–1% prefill impact) and rebalancing to mrr 2 / mamba pool 8 makes FP8 KV +
vision + the full 262144 context fit in 32 GB with ~1.9 GB headroom — zero FP4 KV
caveats, ~2x decode bandwidth per stream. `kv-nvfp4-text-image` remains for those who
need 4 streams on 32 GB (KV 8→5 GB, FP4 caveats apply); if you never send images and want
4 streams, `kv-fp8-text-only` is the long-running text-only high-concurrency profile.

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
2. pulls the pinned SGLang image (digest, with a tag fallback) and the gateway image,
3. downloads the model to `$MODELS_DIR`,
4. generates `.env` (random `SESSION_SECRET`),
5. starts both services and waits for `/health`.

Then open the gateway at `http://localhost:8088`, create a user and a token, and point
your client at it — see [`gateway/opencode-config.md`](gateway/opencode-config.md).

<details>
<summary>Manual online install</summary>

```bash
cp .env.example .env          # edit MODELS_DIR and SESSION_SECRET
pip install -U "huggingface_hub[cli]"
hf download nvidia/Qwen3.8-27B-NVFP4 --local-dir "$MODELS_DIR/Qwen3.8-27B-NVFP4"
docker pull lmsysorg/sglang@sha256:b91d664a8e4825afc16ab831c6035a6c88ac20ef8bd26da4fe2b9813a9f44376
docker pull calciumion/new-api:v1.0.0-rc.36
docker compose --env-file .env -f inference/kv-fp8-text-image.yml up -d
docker compose --env-file .env -f gateway/docker-compose.yml up -d
```
</details>

### Offline (air-gapped machine)

On a **networked** machine:

```bash
make export          # builds bundle/ : both images, the model, checksums (~70 GB)
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

| Argument | `kv-fp8-text-image.yml` (**default**, 32 GB) | `kv-nvfp4-text-image.yml` (32 GB, 4 streams) | `kv-fp8-text-only.yml` (32 GB, 4 streams) |
| --- | --- | --- | --- |
| `--kv-cache-dtype` | `fp8_e4m3` | `nvfp4` | `fp8_e4m3` |
| attention backend | `--attention-backend flashinfer` | `--prefill-attention-backend flashinfer` + `--decode-attention-backend trtllm_mha` | `--attention-backend flashinfer` |
| vision | enabled | enabled | disabled (`language_model_only`) |
| image flags | `--mm-process-config`, `--image-processor-backend pil`, guard `image:128` | same | none |
| mrr / mamba pool / graph bs | **2 / 8 / 2** + `--disable-prefill-cuda-graph` | 4 / 16 / 4 | 4 / 16 / 4 |
| `--mem-fraction-static` | `0.94` | `0.90` | `0.92` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | `expandable_segments:True` | none |
| free VRAM after load (32 GB) | ~1.91 GB | ~2.65 GB | ~0.58 GB |

**Which one?** Start with **FP8 KV + vision** (default): full 262144 context + vision on
32 GB with zero FP4 KV caveats, at 2 concurrent streams (~2x decode bandwidth each).
Pick **NVFP4 KV + vision** only when you need 4 streams on the same card and accept its
**occasional garbled-reasoning instability** ([NVFP4 caveats](#nvfp4-caveats)) — or on a
**24 GB** card, where its smaller KV footprint is the only meaningful option. Pick **FP8
text-only** if you never send images and want 4 streams. On 48 GB+, raise the default profile to mrr 4 /
pool 16 / graph 4 and re-enable the prefill graph (fraction back to 0.90-0.92), then
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

- The mamba state pool caps concurrency at `max_mamba_cache_size // 4` (4 slots budgeted
  per request): pool 16 -> mrr 4, pool 20 -> mrr 5.
- The CUDA-graph decode batch size must cover mrr, or larger batches silently fall back
  to eager mode (~60% slower).
- Default profiles: `kv-fp8-text-image` runs `262144 / mrr 2 / pool 8 / graph 2`
  (prefill graph disabled); the NVFP4 and text-only profiles run `262144 / 4 / 16 / 4`.
- On a 96 GB card you can raise all three together (e.g. mrr 6 / pool 24 / graph 8,
  keeping mrr ≤ pool/4 and graph ≥ mrr). The latch fix has only been validated at 4
  streams, so re-run `tools/concurrency-load.py` (`t1`, `soak`) before trusting higher
  values.

### Mamba slot accounting (why small offline requests saturate the card too)

State slots are charged **per request and per radix path, not per token**: a live request
holds 1 active + up to 2 checkpoints on its path (track interval 256, capped by
`--mamba-max-states-per-path 2`) = 2-3 slots, and admission budgets 4 per request.
Measured: two streams in flight on pool 8 => used 5 / evictable 2 / available 1.
Consequence: an offline "reward-model scoring" loop - 60-1300-token prompts, generation
to a 700-token cap, ~30-50 req/min - needs 10-15 streams by Little's law and pins any
interactive server at full mrr with permanent eviction/reload churn on path states.
Short requests are cheap on KV and expensive on state.
Operational rule: run offline/batch workloads under their **own gateway token** with
**client-side concurrency <= 2** (or off-peak). The fp8v default (mrr 2 / pool 8 =
exactly 2 x the 4-slot admission budget) additionally makes slot exhaustion
structurally unreachable for interactive traffic.

### The scheduler false-latch patch

The pinned SGLang tree has a bug at `scheduler.py:3355`: a chunked-prefill continuation
(which already holds a request row and does not allocate a new one) is counted in
`can_run`, so it is compared against a budget derived from free rows. This double-counts
and sets `batch_is_full` early, capping effective concurrency at `mrr - 1`.

`inference/patches/sched-latch-fix/sitecustomize.py` is mounted at `/patches` and loaded
via `PYTHONPATH` in **all profiles**. It only suppresses the false latch when the real
number of new admits in a pass is below the free rows at the start of the pass — so it
never over-admits and is safe at any `mrr`. **It matches on line number 3355 + function
name** — after any image upgrade, re-check that line or the hook silently no-ops (safe:
it just falls back to `mrr - 1`).

The latch needs an in-flight chunked prefill — a prompt longer than
`chunked_prefill_size` (2048), split across passes. Only then can a second request that
*arrives while that prefill is still running* be held until the first finishes prefill (a
transient `mrr - 1`); the stall lasts as long as the prefill. Two requests arriving in the
same scheduler pass both start, and any prompt <= 2048 tokens never chunks at all, so
neither is affected.

### Context pool budget and the prefill CUDA graph

The pool is profiled as
`pool_bytes = free_after_weights - mem_fraction slack - mm reservation - mamba pool`,
and **CUDA graph capture is not in that formula** - graphs are captured afterwards and
spend the `slack`. So while the pool is pinned by `--max-total-tokens`,
`--mem-fraction-static` moves slack around but not the pool (0.90 vs 0.92: identical);
but when the *profiled* value is the constraint, fraction and graph cost move the pool
directly. Measured 2026-09-11: FP8 KV + vision + mrr 2 profiled to 242337 tokens at
fraction 0.90 with the prefill graph on; disabling that graph (1.19 GB; measured 0-1%
prefill impact - chunked prefill at 2048 tokens is compute-bound, not launch-bound) plus
fraction 0.94 recovered the full 262144 with ~1.9 GB runtime headroom (cold 166K prefill:
87 s, zero retractions). The pool and cold-prefill headroom remain a 1:1 trade
(table below measured with NVFP4 KV / mrr 4, same shape):

| `max-total-tokens` | free VRAM | cold prefill | 4 concurrent |
| --- | --- | --- | --- |
| **262144 (this stack)** | 2.65 GB | ~252K | 4 x 65K |
| 300000 | ~1.9 GB | ~180K | 4 x 75K |
| 320000 | 1.54 GB | ~140K | 4 x 80K |

### Hierarchical cache

`--enable-hierarchical-cache --hicache-ratio 2` keeps an L2 KV cache in host RAM. It
turns a 190K-token re-admission from ~109 s into <1 s with no decode/TTFT regression.
`ratio 2` uses ~17 GB of host RAM; raise it for a larger L2 at the cost of RAM.

### NVFP4 caveats

> **Known occasional instability (observed in production).** This profile emitted intermittent
garbled reasoning — see the first bullet. Prefer the FP8 default unless you need 4
streams on 32 GB, or you are on a **24 GB** card where NVFP4 KV's smaller footprint is
the only meaningful option.

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

- **The NVFP4 KV profile occasionally emits garbled reasoning** (intermittent, observed in
  production; see [NVFP4 caveats](#nvfp4-caveats)). Use it only for 4 streams
  on 32 GB, or on a 24 GB card where it is the only meaningful variant.
- **`page_size` becomes 64 with `trtllm_mha`**, after which the default
  `--mamba-max-states-per-path -1` lets mamba states accumulate per radix path, filling
  the 16-slot pool and deadlocking new allocations (container restart). Both variants set
  `--mamba-max-states-per-path 2` (a zero-VRAM behavioral cap that matches the
  `extra_buffer_lazy` ping-pong design). Always re-run a soak test after changing the KV recipe.
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

### FP8 KV + vision on 32 GB (current default profile, 2026-09-11)

| Check | Result |
| --- | --- |
| boot accounting | KV 8.0 GB + mamba 0.65 GB, ~1.91 GB headroom after graph capture |
| prefill graph off vs on (same stack) | 0 to +1% (1K/14K/70K; TTFT identical) |
| cold 166K prefill | 87.3 s (1905 tok/s), zero retractions |
| single-stream decode @1K | 41-43 tok/s (mrr 2) |
| mamba slots, 2 streams in flight | used 5 / evictable 2 / available 1 (pool 8) |
| images (2 Mpx each) | 1-96 pass, GPU peak flat; 128 rejected by the context window |
| quality spot check | all pass incl. 104K needle |

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
│   ├── kv-fp8-text-image.yml       default: FP8 KV + vision (32 GB, mrr 2)
│   ├── kv-nvfp4-text-image.yml     NVFP4 KV + vision (32 GB, mrr 4)
│   ├── kv-fp8-text-only.yml        FP8 KV, text only, 4 streams (32 GB)
│   └── patches/sched-latch-fix/    scheduler false-latch fix
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
