# llm-infer:spec-f832b5c3 — chained on the hicache build, adds the draft-share patch.
#
# Build context = inference/patches/ (build llm-infer:hicache-f832b5c3 first):
#   docker build -f spec-draft-share/img-f832b5c3/Dockerfile \
#     -t llm-infer:spec-f832b5c3 .
#
# Only inference/kv-fp8-text-image-200k-mtp.yml uses this image; the other
# profiles stay on llm-infer:hicache-f832b5c3, which this is derived from.
#
# Re-anchored onto the v0.5.23 tree (297e54f): the anchor (get_plan_stream +
# def alloc_memory_pool() at eagle_worker_v2.py:241-243) still applies cleanly
# with a +17 offset; init_token_map()/init_lm_head() are still called only from
# alloc_memory_pool(), so the retirement condition is not met upstream.
FROM llm-infer:hicache-f832b5c3

COPY spec-draft-share/img-f832b5c3/patches /tmp/spec-patches

RUN cd /sgl-workspace/sglang \
 && git apply --verbose /tmp/spec-patches/*.patch \
 && rm -rf /tmp/spec-patches \
 && test "$(git status --short | wc -l)" -eq 5 \
 && grep -q "Share the target embedding + lm_head before the KV pool is profiled" \
      python/sglang/srt/speculative/eagle_worker_v2.py \
 && python3 -c "import ast; ast.parse(open('python/sglang/srt/speculative/eagle_worker_v2.py').read())" \
 && echo "spec-draft-share self-check OK"
