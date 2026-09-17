# All-in-one derived image: sched patches + HiCache hybrid-Mamba fix, baked into
# lmsysorg/sglang:latest pinned at index digest
# sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9
# (source tree 0bcd822, v0.5.19 line, built 2026-09-04).
#
# Combines:
#   - inference/patches/sched-latch-fix/img-d6e72886/  (false-latch + LPM wait-boost,
#     baked via 99-sched-latch-fix.pth — see that directory's README);
#   - inference/patches/hicache-mamba-fix/img-d6e72886/patches/  (7-file HiCache
#     hybrid-Mamba fix applied to the /sgl-workspace/sglang checkout — see the
#     parent README).
#
# Build (context = the inference/patches/ directory, NOT this directory):
#   docker build -f hicache-mamba-fix/img-d6e72886/Dockerfile -t llm-infer:hicache-d6e72886 .
#   (the image stays local unless you push it somewhere)
FROM lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9

# --- sched-latch patches (site-packages modules + one-shot .pth loader) ---
COPY sched-latch-fix/img-d6e72886/sched_latch_fix.py /opt/sglang/lib/python3.12/site-packages/sched_latch_fix.py
COPY sched-latch-fix/img-d6e72886/sched_lpm_waitfix.py /opt/sglang/lib/python3.12/site-packages/sched_lpm_waitfix.py
RUN printf 'import sched_latch_fix\nimport sched_lpm_waitfix\n' > /opt/sglang/lib/python3.12/site-packages/99-sched-latch-fix.pth

# --- HiCache hybrid-Mamba fix (source patches on the editable checkout) ---
COPY hicache-mamba-fix/img-d6e72886/patches /tmp/hicache-patches
RUN cd /sgl-workspace/sglang \
 && git apply --verbose /tmp/hicache-patches/*.patch \
 && rm -rf /tmp/hicache-patches

# Build-time self-check: exactly the 7 expected files modified, every new marker
# present. A context mismatch upstream fails the build here, not at runtime.
RUN cd /sgl-workspace/sglang \
 && test "$(git status --short | wc -l)" -eq 7 \
 && grep -q "Write-through backup for chunked prefills" python/sglang/srt/mem_cache/unified_radix_cache.py \
 && grep -q "loaded_host_hit_length" python/sglang/srt/managers/schedule_batch.py \
 && grep -q "loaded_host_hit_length" python/sglang/srt/managers/schedule_policy.py \
 && grep -q "_try_alloc_mamba_slot" python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py \
 && grep -q "radix_cache_aux_alloc_failed_total" python/sglang/srt/observability/metrics_collector.py \
 && grep -q "SGLANG_HICACHE_MAMBA_SIZE_GB" python/sglang/srt/environ.py \
 && grep -q "SGLANG_HICACHE_MAMBA_SIZE_GB" python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py
