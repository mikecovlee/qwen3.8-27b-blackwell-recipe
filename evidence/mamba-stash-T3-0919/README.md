# T3 stash-crash gate — v0.5.20 build (2026-09-19)

First T3 run on the migrated stack: image `llm-infer:hicache-06e4f2ed`
(v0.5.20 tree `94602c9`, sched-latch re-anchored at scheduler.py:3887,
hicache P1/P3/C carried, P2 retired as upstream-absorbed, LPM waitfix dropped),
profile `inference/kv-fp8-text-image-v0520-hrrn.yml`
(`--schedule-policy hrrn`, sizing identical to mainline).

`verify-mamba-stash.py --expect safe`, 2 trials: **PASS both** — 98K/98K/153K
stash-pressure sequence admitted with 0 crash lines, 0 container restarts
(`result.json`, per-trial logs in `trial-*/`).

Same-session gates: `verify-hicache-thrash.py` PASS (244K cold prefill 124 s →
evicted reload 35 s with `load_back` +136K → continuation 0.4 s, 0 retractions);
`verify-scheduling-patches.py` T1 latch PASS behaviorally (B decodes at A's
prefill end, 34 s before A's turn end); T2 fails by design — the waitfix module
is not baked in this build. HRRN aging observed in the T2 scenario: the cold
waiter was admitted before one hot overtaker (TTFT 37.8 s ≈ 1 round vs the
~2-round unboosted expectation).
