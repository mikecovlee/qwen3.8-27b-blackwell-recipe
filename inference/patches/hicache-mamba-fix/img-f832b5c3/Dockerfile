# All-in-one derived image: HiCache hybrid-Mamba fix, baked into
# lmsysorg/sglang:v0.5.23 pinned at index digest
# sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f
# (source tree 297e54f, v0.5.23 line, built 2026-11-09).
#
# Contains only the hicache-mamba-fix patch set (4 files):
#   - P1 (chunked write-through backup) stays RETIRED: the first-class
#     chunked-prefill stash + flush_pending_backups() pipeline is still
#     upstream.
#   - P2 stays RETIRED: absorbed upstream as Req.host_loaded_length +
#     materialized_host_hit_len().
#   - P3 (mamba skip-vs-assert) PORTED onto the v0.5.23 tree: upstream #36770
#     is still open, but _alloc_mamba_slot gained a req parameter and a lazy
#     extra-buffer lock-release step, so the hunk was rewritten rather than
#     re-anchored. The three donate-path callers now use _try_alloc_mamba_slot
#     (skip caching on failure); the external-linker LOAD path keeps the fatal
#     _alloc_mamba_slot.
#   - C (SGLANG_HICACHE_MAMBA_SIZE_GB host mamba pool knob) re-applied with
#     offsets; still dormant unless the env is set. REQUIRED on v0.5.22 by the
#     host mamba pool overflow (evidence/hicache-v0522-regression-1020);
#     re-check via T4 on v0.5.23 before dropping.
#
# vs the e3a4f060 (v0.5.22) build:
#   - sched-latch-fix is NOT baked: upstream PR #44401 ("Count a request's
#     held req_to_token row once at admission", merged 2026-10-15) fixed the
#     double-counting this patch suppressed — get_num_allocatable_reqs now
#     adds num_held_rows back to the pool term. Do NOT re-add.
#
# Build (context = the inference/patches/ directory, NOT this directory):
#   docker build -f hicache-mamba-fix/img-f832b5c3/Dockerfile -t llm-infer:hicache-f832b5c3 .
#   (the image stays local unless you push it somewhere)
# Smoke (GPU-less):
#   docker run --rm --network=none --entrypoint python3 llm-infer:hicache-f832b5c3 \
#     -c "import sglang.srt.managers.scheduler"
FROM lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f

# --- HiCache hybrid-Mamba fix (source patches on the editable checkout) ---
COPY hicache-mamba-fix/img-f832b5c3/patches /tmp/hicache-patches
RUN cd /sgl-workspace/sglang \
 && git apply --verbose /tmp/hicache-patches/*.patch \
 && rm -rf /tmp/hicache-patches

# Build-time self-check: exactly the 4 expected files modified, every new marker
# present, the upstream-absorbed P1/P2 mechanisms are NOT reintroduced, and the
# retired sched-latch module is NOT present. A context mismatch upstream fails
# the build here, not at runtime.
RUN cd /sgl-workspace/sglang \
 && test "$(git status --short | wc -l)" -eq 4 \
 && grep -q "_try_alloc_mamba_slot" python/sglang/srt/mem_cache/unified_cache/components/mamba.py \
 && grep -q "radix_cache_aux_alloc_failed_total" python/sglang/srt/observability/metrics_collector.py \
 && grep -q "SGLANG_HICACHE_MAMBA_SIZE_GB" python/sglang/srt/environ.py \
 && grep -q "SGLANG_HICACHE_MAMBA_SIZE_GB" python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py \
 && ! grep -rq "loaded_host_hit_length" python/sglang/srt/managers/schedule_batch.py python/sglang/srt/managers/schedule_policy.py \
 && grep -q "def materialized_host_hit_len" python/sglang/srt/managers/schedule_batch.py \
 && grep -q "num_held_rows" python/sglang/srt/managers/scheduler.py \
 && test ! -f /opt/sglang/lib/python3.12/site-packages/sched_latch_fix.py
