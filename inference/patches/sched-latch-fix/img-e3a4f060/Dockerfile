# Derived image that bakes the sched-latch false-latch patch into
# lmsysorg/sglang:v0.5.22 pinned at index digest
# sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae
# (source tree baa9bef, v0.5.22 line, built 2026-10-19).
#
# v0.5.22 note: the double-count false latch SURVIVED this release — v0.5.22
# only added the #41841/#41872 empty-batch latch fix plus a pre-loop
# allocatable<=0 guard; the gate at scheduler.py:4098-4103 is unchanged, so
# the patch is re-anchored to scheduler.py:4103 (NOT retired). Re-run the T1
# acceptance (48K chunked prefill + mid-prefill arrival) before relying on it.
#
# Build (context = the sched-latch-fix/ directory):
#   docker build -f img-e3a4f060/Dockerfile -t sglang-latchfix:e3a4f060 .
#   (the image stays local unless you push it somewhere)
FROM lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae

COPY img-e3a4f060/sched_latch_fix.py /opt/sglang/lib/python3.12/site-packages/sched_latch_fix.py
RUN printf 'import sched_latch_fix\n' > /opt/sglang/lib/python3.12/site-packages/99-sched-latch-fix.pth
