# All-in-one derived image: sched-latch patch + HiCache hybrid-Mamba fix, baked
# into lmsysorg/sglang:v0.5.22 pinned at index digest
# sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae
# (source tree baa9bef, v0.5.22 line, built 2026-10-19).
#
# Combines:
#   - inference/patches/sched-latch-fix/img-e3a4f060/  (false-latch, re-anchored
#     to scheduler.py:4103 — the double counting survives v0.5.22, see that
#     directory's Dockerfile note; baked via 99-sched-latch-fix.pth),
#   - inference/patches/hicache-mamba-fix/img-e3a4f060/patches/  (4-file
#     HiCache hybrid-Mamba fix re-anchored onto the v0.5.22 tree).
#
# vs the b1259f3e (v0.5.21) build:
#   - P1 (chunked write-through backup) stays RETIRED: v0.5.22 still ships the
#     first-class chunked-prefill stash + flush_pending_backups() pipeline, and
#     the Rust TreeCore has no node_by_id().
#   - P3 (mamba skip-vs-assert) re-applied with offsets (upstream #36770 still
#     open as of v0.5.22).
#   - C (SGLANG_HICACHE_MAMBA_SIZE_GB host mamba pool knob) re-anchored:
#     hybrid_pool_assembler.py restructured the host-size split into an
#     `if get_memory().hicache_size > 0:` block with a QSA kv re-scale step;
#     the override now sits right after that block in both stacks.
#     Still a switchable fallback (dormant unless the env is set); the
#     aggressive path relies on upstream hicache_auto_size.py.
#
# Build (context = the inference/patches/ directory, NOT this directory):
#   docker build -f hicache-mamba-fix/img-e3a4f060/Dockerfile -t llm-infer:hicache-e3a4f060 .
#   (the image stays local unless you push it somewhere)
# Smoke (GPU-less, expect the sched-latch self-check to log
# 'anchor verified ...:4103' + 'installed'):
#   docker run --rm --network=none --entrypoint python3 llm-infer:hicache-e3a4f060 \
#     -c "import sglang.srt.managers.scheduler"
FROM lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae

# --- sched-latch patch (site-packages module + one-shot .pth loader) ---
COPY sched-latch-fix/img-e3a4f060/sched_latch_fix.py /opt/sglang/lib/python3.12/site-packages/sched_latch_fix.py
RUN printf 'import sched_latch_fix\n' > /opt/sglang/lib/python3.12/site-packages/99-sched-latch-fix.pth

# --- HiCache hybrid-Mamba fix (source patches on the editable checkout) ---
COPY hicache-mamba-fix/img-e3a4f060/patches /tmp/hicache-patches
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
