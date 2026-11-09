# llm-infer:nvfp4-f832b5c3 — chained on the hicache build, adds the NVFP4
# host-transport fix (port of upstream PR #46452, open on main as of 2026-11-09).
#
# Why: NVFP4 KV stores packed fp4 payload plus per-block scales in separate
# tensors (KVScaleLayout.TOKEN). v0.5.23's MHA host pool accepts only
# MXFP8_INTERLEAVED scales and fails startup with "MHA host transfers do not
# support KV scale layout 'token'"; v0.5.21 silently dropped the scales on
# write-back/load-back (upstream #43239/#36121 silent-corruption bug).
# This port (4 files: device_pool_info, memory_pool, pool_host/mha,
# pool_host/mha_fp4) adds MHATokenToKVPoolFP4Host: byte-width payload rows +
# page-granular scale transport on the kernel/page_first path.
#
# Build context = inference/patches/ (build llm-infer:hicache-f832b5c3 first):
#   docker build -f nvfp4-host-transport/img-f832b5c3/Dockerfile \
#     -t llm-infer:nvfp4-f832b5c3 .
#
# Only inference/kv-nvfp4-text-image.yml uses this image.
FROM llm-infer:hicache-f832b5c3

COPY nvfp4-host-transport/img-f832b5c3/patches /tmp/nvfp4-patches

RUN cd /sgl-workspace/sglang \
 && git apply --verbose /tmp/nvfp4-patches/*.patch \
 && rm -rf /tmp/nvfp4-patches \
 && test "$(git status --short | wc -l)" -eq 8 \
 && grep -q "MHATokenToKVPoolFP4Host" python/sglang/srt/mem_cache/pool_host/mha_fp4.py \
 && grep -q "MHATokenToKVPoolFP4Host" python/sglang/srt/mem_cache/pool_host/mha.py \
 && python3 -c "import ast; ast.parse(open('python/sglang/srt/mem_cache/pool_host/mha_fp4.py').read())" \
 && echo "nvfp4-host-transport self-check OK"
