# spec-draft-share

Shares the target model's `embed_tokens` + `lm_head` with the NEXTN/MTP draft
**before** the KV pool is profiled, so the draft's throwaway copies do not shrink
the context window.

## The bug (stock v0.5.21, tree `e00930c`)

In `speculative/eagle_worker_v2.py`, `EagleDraftWorker.__init__` builds the draft,
which allocates its own bf16 `embed_tokens` + `lm_head` — 2 × 2.54 GB at vocab
248320. The checkpoint's `mtp.*` tensors contain neither, so both are always
overwritten later by `set_embed_and_head()`.

But the KV pool is profiled immediately after the draft load:

```
draft Load weight end. type=Qwen3_5ForCausalLMMTP, mem usage=5.53 GB
max_total_tokens=262144 is larger than the profiled value 97068
...
Capture target verify CUDA graph begin ... avail mem=5.44 GB   # ~4.1 GB just freed
```

`init_token_map()` + `init_lm_head()` only run from `alloc_memory_pool()`, which the
scheduler calls *after* the pool exists. So the pool is sized against a footprint
that is freed moments later: **5.24 GB left idle, pool 97068 tokens (37% of the
262144 context)**.

## The patch

9 insertions, hunk `@@ -223,6 +223,15 @@ class EagleDraftWorker`, at the end of
`__init__`:

```python
if get_parallel().pp_size == 1:
    self.init_token_map()
    self.init_lm_head()
```

The later calls in `alloc_memory_pool()` are left in place: `set_embed_and_head()`
deletes and re-assigns the *same* target tensors, so a second call is a no-op.
Guarded on `pp_size == 1` because under PP `resolve_draft_embed_and_head()` may fall
back to `load_draft_embedding_from_checkpoint()`, which needs the draft build
context.

Inert without `--speculative-algorithm` (the class is never constructed), so the
other three profiles are unaffected.

## Measured effect (32 GB, Qwen3.8-27B-NVFP4)

| config | KV pool | headroom | stability |
|---|---|---|---|
| stock + spec | 97068 (37%) | 5.24 GB idle | — |
| patched, frac 0.94 | 223678 (85%) | 1.18 GB | OOM-crashed under verify-mamba-stash |
| patched, frac 0.92 | **204519 (78%)** | 1.79 GB | T3 PASS, 0 restarts |

decode 42.3 → 61.8–62.6 tok/s (1.47×; accept length 1.73–1.93 of a max 2),
48.9 @64K, 43.0 @128K. Prefill @4K unchanged (5611 vs 5489 tok/s); @2K regresses
31% (TTFT 0.37 → 0.54 s) because draft extend is a fixed per-chunk cost.

## Build

```bash
cd inference/patches
docker build -f spec-draft-share/img-b1259f3e/Dockerfile -t llm-infer:spec-b1259f3e .
```

Chained on `llm-infer:hicache-b1259f3e`, so it also carries sched-latch + hicache
P3 (5 modified files total; the Dockerfile self-check asserts exactly 5 plus a
syntax parse).

## Re-anchoring on a new base

1. Confirm `git status` is clean in `/sgl-workspace/sglang`.
2. Find the end of `EagleDraftWorker.__init__`: the line
   `self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)` followed
   by a blank line and `def alloc_memory_pool(`. The same `get_plan_stream` line
   also appears in `EAGLEWorkerV2.__init__` — the two-line anchor including
   `def alloc_memory_pool(` is what makes it unique.
3. `git apply --check`, then a GPU-less smoke:
   `docker run --rm --entrypoint python3 <image> -c 'import sglang.srt.managers.scheduler'`.
4. Run the acceptance gates on an isolated twin before pointing a profile at it.
   T1/T4 size their prompts for a 262144 pool, so scale them down:
   `verify-scheduling-patches.py --a-reps 6000` (~156K tokens) and
   `verify-hicache-thrash.py --a-chars 180000 --bc-chars 70000`.

## Retirement

Retire when upstream profiles the KV pool *after* the draft shares the target's
embed/lm_head — i.e. when `init_lm_head()` moves out of `alloc_memory_pool()`, or
the draft stops allocating weights it is about to discard.
