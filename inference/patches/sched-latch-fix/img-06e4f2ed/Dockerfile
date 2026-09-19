# Derived image that bakes the sched-latch false-latch patch into
# lmsysorg/sglang:latest pinned at index digest
# sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f
# (source tree 94602c9, v0.5.20 line, built 2026-09-18).
#
# v0.5.20 note: the LPM wait-boost companion is NOT baked here. Upstream shipped
# `--schedule-policy hrrn` (aging-based, mean TTFT -69% vs FCFS on their trace),
# so we observe HRRN in production instead of the local waitfix. Re-add
# sched_lpm_waitfix.py only if HRRN disappoints.
#
# Build (context = the sched-latch-fix/ directory):
#   docker build -f img-06e4f2ed/Dockerfile -t sglang-latchfix:06e4f2ed .
#   (the image stays local unless you push it somewhere)
FROM lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f

COPY img-06e4f2ed/sched_latch_fix.py /opt/sglang/lib/python3.12/site-packages/sched_latch_fix.py
RUN printf 'import sched_latch_fix\n' > /opt/sglang/lib/python3.12/site-packages/99-sched-latch-fix.pth
