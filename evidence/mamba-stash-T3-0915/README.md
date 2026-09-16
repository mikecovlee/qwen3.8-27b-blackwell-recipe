# Mamba stash crash reproduction & E10 fix verification (2026-09-15)

Evidence for the 2026-09-15 10:29 CST production crash (`AssertionError: Can not alloc
mamba cache` in `_alloc_mamba_slot` during `stash_chunked_request`) and the E10 fix
(`--mamba-radix-cache-strategy extra_buffer` + `--max-mamba-cache-size 10`).

Root cause: `extra_buffer_lazy` admission coefficient 2 < actual peak 3 slots/req
(active + ping-pong track + stash-donated); upstream FIXME at `schedule_policy.py:794`
and unit test `test_mamba_donated_alloc_ratio.py` pin this as fail-loud by design
(peak = N own + N locked + 1 donated; ratio 2 without an evictable victim must assert).
Full write-up: `local/LESSONS-0908.md` §十一.

## Layout

| Path | Contents |
| --- | --- |
| `crash-arm/` | T3 run on lazy/pool-8 (`--expect crash`): RESULT FAIL — assert not reproduced on a clean twin tree (donated checkpoints are evictable leaves), but the timeline proves the undercount: L admitted at `mamba_avail=2`, stash forced an eviction. Raw twin container log was lost on recreate; `result.json` + `trial-*/timeline.jsonl` carry the key data. |
| `safe-arm/` | T3 run on extra_buffer/pool-10 (`--expect safe`): **PASS** — L (153K) full chunked prefill in 75 s, admitted right after M1 finished, `avail` constant 2, used peak 8/10, `evict_for_alloc`=0, no assert, zero restarts. |
| `twin-e10-full.log` | Full twin container log for the E10 arm (boot accounting + T3 safe + T1/T2 + regression suite). |

## Key numbers (E10 twin, vs baselines)

| Check | Result | Baseline |
| --- | --- | --- |
| KV pool tokens (boot log) | 262144 (unchanged) | 262144 — context loss zero |
| Memory pool end avail | 2.43 GB | 2.59 GB (delta 0.16 GB = +2 mamba slots, as predicted) |
| T1 latch / T2 boost | PASS / PASS (C TTFT 24.9 s, boost-log 0→2) | PASS |
| single-stream decode 1K / 64K | 43.1 / 38.8 tok/s | 42 / 38 |
| dual-stream bs2 graph median (engine side) | 69.7 tok/s (eager 0) | 69-78 |
| cold 166K prefill TTFT | 87.1 s | ~87 s |
| cache 2nd hit | 2176/2190 = 99.4 % | ~99 % |
| 150K×2 squeeze | retract 0 / assert 0 (2nd queues by design, 300K > 262K pool) | zero retraction |

Production cutover 23:48 CST: rendered-config diff gate passed (only the two E10 tokens
differ), `verify-scheduling-patches` T1/T2 PASS on :8080, cache-report 2112/2159 cached,
gateway clean.

## Reproduce

```sh
# twin (from oss/, production stopped or off-peak; same GPU — mutually exclusive)
MAMBA_STRATEGY=extra_buffer_lazy MAMBA_POOL=8 docker compose -p inference \
  --project-directory . -f inference/kv-fp8-text-image.yml \
  -f inference/tools/acceptance/twin-override.yml up -d
python3 inference/tools/acceptance/verify-mamba-stash.py --expect crash \
  --outdir evidence/mamba-stash-T3-0915/crash-arm
# E10 arm: recreate with default env (extra_buffer/10), then rerun with --expect safe
```

Positive-control note: the crash arm does not deterministically fire the assert on a
clean tree — it needs an aged radix tree with locked/forked paths plus a hicache mamba
backup pinning window ("no evictable victim"). The authoritative positive controls are
the production incident itself and the upstream unit test cited above. The crash arm's
value is the live demonstration of the coefficient-2 undercount (admission at avail=2).
