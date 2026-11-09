# llm-infer:spec-b1259f3e — chained on the hicache build, adds the draft-share patch.
#
# Build context = inference/patches/ (build llm-infer:hicache-b1259f3e first):
#   docker build -f spec-draft-share/img-b1259f3e/Dockerfile \
#     -t llm-infer:spec-b1259f3e .
#
# Only inference/kv-fp8-text-image-200k-mtp.yml uses this image; the other three
# profiles stay on llm-infer:hicache-b1259f3e, which this is derived from.
FROM llm-infer:hicache-b1259f3e

COPY spec-draft-share/img-b1259f3e/patches /tmp/spec-patches

RUN cd /sgl-workspace/sglang \
 && git apply --verbose /tmp/spec-patches/*.patch \
 && rm -rf /tmp/spec-patches \
 && test "$(git status --short | wc -l)" -eq 5 \
 && grep -q "Share the target embedding + lm_head before the KV pool is profiled" \
      python/sglang/srt/speculative/eagle_worker_v2.py \
 && python3 -c "import ast; ast.parse(open('python/sglang/srt/speculative/eagle_worker_v2.py').read())" \
 && echo "spec-draft-share self-check OK"
