# sglang-0523-upgrade-1109 — SGLang v0.5.22 → v0.5.23 upgrade acceptance

Date: 2026-11-09.  Build: `llm-infer:hicache-f832b5c3` (base
`lmsysorg/sglang@sha256:f832b5c3…`, tree 297e54f, v0.5.23).
Isolated twin `llm-infer-test` on :8099 (compose project `inference-test`,
same profile args), production `llm-infer` stopped for the window.

## Gates

| Gate | Result | Evidence |
|---|---|---|
| T1 latch (verify-scheduling-patches.py) | **PASS** — B decode@168.5 = A prefill end, A turn-done@202.6; latch-log 0→0 (module retired, upstream #44401 fix verified behaviorally) | `t1.log` |
| T3 mamba stash (verify-mamba-stash.py --expect safe, 2 trials) | **PASS** safe/safe, zero crash lines, zero restarts | `t3.log`, `mamba-stash-T3-20261109-102651/` |
| T4 hicache thrash WITH `SGLANG_HICACHE_MAMBA_SIZE_GB=8.5` | **PASS** — D=0.4 s, load_back kv +137344, retract=0 assert=0 | `t4-with-env.log` |
| T4 WITHOUT the env (A/B for the C knob) | **REPLAY** — D=124.5 s, load_back +0 → env STILL REQUIRED on v0.5.23 | `t4-no-env.log` |

## Conclusions

- sched-latch-fix retired: upstream PR #44401 (merged 2026-10-15) fixed the
  double count; T1 passes with no patch module loaded.
- hicache C (`SGLANG_HICACHE_MAMBA_SIZE_GB=8.5`) remains load-bearing on
  v0.5.23 — keep the env in `kv-fp8-text-image.yml`.
- hicache P3 (ported to `_alloc_mamba_slot(req)`) and spec-draft-share
  (rebased, +17 offset) ship in the new images; P3's skip path is not
  exercised by these gates (it degrades gracefully, see patch README).

## 200k-mtp profile (llm-infer:spec-f832b5c3, same window)

Scaled gates for the 204800 pool (`--a-reps 6000`, `--a-chars 180000
--bc-chars 70000`), twin with `kv-fp8-text-image-200k-mtp.yml`:

| Gate | Result | Evidence |
|---|---|---|
| T1 latch scaled | **PASS** — B@75.8 = A prefill end, turn 17 s | `t1-mtp.log` |
| T3 mamba stash | **PASS** safe/safe, zero restarts | `t3-mtp.log`, `mamba-stash-T3-20261109-113404/` |
| T4 scaled WITHOUT env | **PASS** (D=0.4 s, +89536) — the scaled gate fits the default pool | `t4-mtp-noenv.log` |
| T4 scaled WITH env 8.5 | **PASS** (D=0.4 s, +89536) | `t4-mtp-with-env.log` |

Note: the scaled gate passing without env does NOT prove the full 204800
workload is safe without it (26 chunk anchors at gate size vs 33 at full
context; the default profile's full-size A/B REPLAYs without env on the same
tree). The env is therefore kept in the mtp profile — consistent with its
"inherited from the default" contract, and matching the default profile's
2026-10-20 finding.

## text-only profile (llm-infer:hicache-f832b5c3, same window)

High-concurrency variant (mrr 4 / pool 16 / graph 4, `extra_buffer_lazy`,
262144 pool), default gate sizes:

| Gate | Result | Evidence |
|---|---|---|
| T1 latch | **PASS** — B@163.3 = A prefill end, turn 34 s | `t1-textonly.log` |
| T3 mamba stash | **PASS** safe/safe, zero restarts | `t3-textonly.log`, `mamba-stash-T3-20261109-134819/` |
| T4 WITHOUT env | **REPLAY** (D=119.6 s, load_back +0) | `t4-textonly-noenv.log` |
| T4 WITH env 8.5 | **PASS** (D=0.4 s, +135296) | `t4-textonly-with-env.log` |

Switched to `hicache-f832b5c3` + `SGLANG_HICACHE_MAMBA_SIZE_GB=8.5` on
2026-11-09 (same C-knob requirement as the default profile).

## nvfp4 profile (llm-infer:nvfp4-f832b5c3, same window)

v0.5.23 startup initially died in `MHATokenToKVPoolHost.__init__`
(`pool_host/mha.py:166`): `MHA host transfers do not support KV scale layout
'token'`. NVFP4 KV stores per-block scales in separate tensors (`TOKEN`
layout); v0.5.23's MHA host pool accepts only `MXFP8_INTERLEAVED` scales.
History: on v0.5.21 this combination SILENTLY dropped the scales on
write-back/load-back (upstream #43239/#36121 silent-corruption bug) — v0.5.23
fails fast instead, which is strictly safer.

No upstream fix is merged (#36121, #43240, #46452 all open), so **PR #46452
("preserve NVFP4 payload and scales across host transfers") was ported as a
local patch set** `inference/patches/nvfp4-host-transport/img-f832b5c3/`
(4 files: device_pool_info, memory_pool, pool_host/mha, pool_host/mha_fp4 —
adds `MHATokenToKVPoolFP4Host` with byte-width payload rows + page-granular
scale transport on the kernel/page_first path). Applies cleanly on top of
the hicache chain; image built + self-checked 2026-11-09.

Gates (isolated twin, default gate sizes, 262144 pool):

| Gate | Result | Evidence |
|---|---|---|
| T1 latch | **PASS** — B@168.5 = A prefill end, turn 30 s | `t1-nvfp4.log` |
| T3 mamba stash | **PASS** safe/safe, zero restarts | `t3-nvfp4.log`, `mamba-stash-T3-20261109-143732/` |
| T4 WITHOUT env | **REPLAY** (D=125.2 s, load_back +0) | `t4-nvfp4-noenv.log` |
| T4 WITH env 8.5 | **PASS** (D=0.3 s, +135296) | `t4-nvfp4-with-env.log` |

Note: the port fixes transport correctness (upstream's byte-exact battery is
in #46452); T4 reload proves the load-back path is exercised end-to-end. When
#46452 (or #43240) merges upstream, drop the local patch set on the next bump.

## Switch

`oss/inference/kv-fp8-text-image.yml` → `llm-infer:hicache-f832b5c3` on
2026-11-09 after all gates green.  Production restarted 10:47, smoke OK,
zero errors/restarts.
`oss/inference/kv-fp8-text-image-200k-mtp.yml` → `llm-infer:spec-f832b5c3`
(plus the env addition above) on 2026-11-09 after its scaled gates green;
default-profile production restarted 12:0x, smoke OK.
`oss/inference/kv-fp8-text-only.yml` → `llm-infer:hicache-f832b5c3` + env 8.5
on 2026-11-09 after its gates green.
`oss/inference/kv-nvfp4-text-image.yml` → `llm-infer:nvfp4-f832b5c3` + env 8.5
on 2026-11-09 after its gates green; default-profile production restarted
14:5x, smoke OK. All four profiles now run the v0.5.23 line.
