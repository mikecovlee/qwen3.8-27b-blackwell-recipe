# Derived image that bakes the sched-latch false-latch patch into
# lmsysorg/sglang:v0.5.21 pinned at index digest
# sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469
# (source tree e00930c, v0.5.21 line, built 2026-10-01).
#
# v0.5.21 note: same false-latch as v0.5.20 (anchor scheduler.py:3976). The LPM
# wait-boost companion stays retired in favor of upstream `--schedule-policy hrrn`.
#
# Build (context = the sched-latch-fix/ directory):
#   docker build -f img-b1259f3e/Dockerfile -t sglang-latchfix:b1259f3e .
#   (the image stays local unless you push it somewhere)
FROM lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469

COPY img-b1259f3e/sched_latch_fix.py /opt/sglang/lib/python3.12/site-packages/sched_latch_fix.py
RUN printf 'import sched_latch_fix\n' > /opt/sglang/lib/python3.12/site-packages/99-sched-latch-fix.pth
