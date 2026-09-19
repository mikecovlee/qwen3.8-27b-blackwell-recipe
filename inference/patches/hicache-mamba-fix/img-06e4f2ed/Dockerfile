# All-in-one derived image: sched-latch patch + HiCache hybrid-Mamba fix, baked
# into lmsysorg/sglang:latest pinned at index digest
# sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f
# (source tree 94602c9, v0.5.20 line, built 2026-09-18).
#
# Combines:
#   - inference/patches/sched-latch-fix/img-06e4f2ed/  (false-latch only — the
#     LPM waitfix retired 2026-09-19 in favor of upstream --schedule-policy
#     hrrn; baked via 99-sched-latch-fix.pth),
#   - inference/patches/hicache-mamba-fix/img-06e4f2ed/patches/  (5-file HiCache
#     hybrid-Mamba fix re-anchored onto the v0.5.20 tree — see parent README).
#
# vs the d6e72886 (v0.5.19) build: the honest-host-hit metric (loaded_host_hit_length)
# is RETIRED — v0.5.20 absorbed it upstream as host_loaded_length +
# materialized_host_hit_len(). The mamba patch follows the upstream file rename
# components/mamba_component.py -> components/mamba.py.
#
# Build (context = the inference/patches/ directory, NOT this directory):
#   docker build -f hicache-mamba-fix/img-06e4f2ed/Dockerfile -t llm-infer:hicache-06e4f2ed .
#   (the image stays local unless you push it somewhere)
FROM lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f

# --- sched-latch patch (site-packages module + one-shot .pth loader) ---
COPY sched-latch-fix/img-06e4f2ed/sched_latch_fix.py /opt/sglang/lib/python3.12/site-packages/sched_latch_fix.py
RUN printf 'import sched_latch_fix\n' > /opt/sglang/lib/python3.12/site-packages/99-sched-latch-fix.pth

# --- HiCache hybrid-Mamba fix (source patches on the editable checkout) ---
COPY hicache-mamba-fix/img-06e4f2ed/patches /tmp/hicache-patches
RUN cd /sgl-workspace/sglang \
 && git apply --verbose /tmp/hicache-patches/*.patch \
 && rm -rf /tmp/hicache-patches

# Build-time self-check: exactly the 5 expected files modified, every new marker
# present, and the upstream-absorbed P2 mechanism is NOT reintroduced. A context
# mismatch upstream fails the build here, not at runtime.
RUN cd /sgl-workspace/sglang \
 && test "$(git status --short | wc -l)" -eq 5 \
 && grep -q "Write-through backup for chunked prefills" python/sglang/srt/mem_cache/unified_radix_cache.py \
 && grep -q "_try_alloc_mamba_slot" python/sglang/srt/mem_cache/unified_cache/components/mamba.py \
 && grep -q "radix_cache_aux_alloc_failed_total" python/sglang/srt/observability/metrics_collector.py \
 && grep -q "SGLANG_HICACHE_MAMBA_SIZE_GB" python/sglang/srt/environ.py \
 && grep -q "SGLANG_HICACHE_MAMBA_SIZE_GB" python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py \
 && ! grep -rq "loaded_host_hit_length" python/sglang/srt/managers/schedule_batch.py python/sglang/srt/managers/schedule_policy.py \
 && grep -q "def materialized_host_hit_len" python/sglang/srt/managers/schedule_batch.py
