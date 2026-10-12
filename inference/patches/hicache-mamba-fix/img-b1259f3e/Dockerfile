# All-in-one derived image: sched-latch patch + HiCache hybrid-Mamba fix, baked
# into lmsysorg/sglang:v0.5.21 pinned at index digest
# sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469
# (source tree e00930c, v0.5.21 line, built 2026-10-01).
#
# Combines:
#   - inference/patches/sched-latch-fix/img-b1259f3e/  (false-latch only — baked
#     via 99-sched-latch-fix.pth),
#   - inference/patches/hicache-mamba-fix/img-b1259f3e/patches/  (4-file HiCache
#     hybrid-Mamba fix re-anchored onto the v0.5.21 tree — see parent README).
#
# vs the 06e4f2ed (v0.5.20) build:
#   - P1 (chunked write-through backup, unified_radix_cache.py) is RETIRED:
#     v0.5.21 ships a first-class chunked-prefill stash + flush_pending_backups()
#     backup pipeline (scheduler.stash_chunked_request -> maybe_cache_unfinished_req
#     (chunked=True) -> cache_unfinished_req(chunked=True)), and the Rust TreeCore
#     has no node_by_id() so the old hunk cannot run anyway.
#   - P3 (mamba skip-vs-assert) re-anchored: components/mamba.py:509.
#   - C (SGLANG_HICACHE_MAMBA_SIZE_GB host mamba pool knob) re-anchored and kept
#     as a switchable fallback; the aggressive path relies on upstream
#     hicache_auto_size.py instead (active only when --hicache-ratio is unset).
#
# Build (context = the inference/patches/ directory, NOT this directory):
#   docker build -f hicache-mamba-fix/img-b1259f3e/Dockerfile -t llm-infer:hicache-b1259f3e .
#   (the image stays local unless you push it somewhere)
FROM lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469

# --- sched-latch patch (site-packages module + one-shot .pth loader) ---
COPY sched-latch-fix/img-b1259f3e/sched_latch_fix.py /opt/sglang/lib/python3.12/site-packages/sched_latch_fix.py
RUN printf 'import sched_latch_fix\n' > /opt/sglang/lib/python3.12/site-packages/99-sched-latch-fix.pth

# --- HiCache hybrid-Mamba fix (source patches on the editable checkout) ---
COPY hicache-mamba-fix/img-b1259f3e/patches /tmp/hicache-patches
RUN cd /sgl-workspace/sglang \
 && git apply --verbose /tmp/hicache-patches/*.patch \
 && rm -rf /tmp/hicache-patches

# Build-time self-check: exactly the 4 expected files modified, every new marker
# present, and the upstream-absorbed P1/P2 mechanisms are NOT reintroduced. A
# context mismatch upstream fails the build here, not at runtime.
RUN cd /sgl-workspace/sglang \
 && test "$(git status --short | wc -l)" -eq 4 \
 && grep -q "_try_alloc_mamba_slot" python/sglang/srt/mem_cache/unified_cache/components/mamba.py \
 && grep -q "radix_cache_aux_alloc_failed_total" python/sglang/srt/observability/metrics_collector.py \
 && grep -q "SGLANG_HICACHE_MAMBA_SIZE_GB" python/sglang/srt/environ.py \
 && grep -q "SGLANG_HICACHE_MAMBA_SIZE_GB" python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py \
 && ! grep -rq "loaded_host_hit_length" python/sglang/srt/managers/schedule_batch.py python/sglang/srt/managers/schedule_policy.py \
 && grep -q "def materialized_host_hit_len" python/sglang/srt/managers/schedule_batch.py
