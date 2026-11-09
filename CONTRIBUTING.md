# Contributing / maintenance conventions

## One source of truth

Everything generic lives in this repo: patches, scheduler builds, probes, acceptance
tools, profiles, docs. A deployment checkout on a machine keeps only machine
bindings (absolute paths, locally built image tags, rollback backups) and private
operations scripts. If you maintain a deployment, contribute fixes back here rather
than forking them into local copies.

## Changing the serve image

1. Run the quant gate probe on the new image (GPU required, one-shot):
   `tools/probe-bmm-fp8.py` — FP8 kernels must match bit-for-bit against the cutlass
   reference on your GPU generation.
2. Add a new `img-<first8-of-digest>/` build under each patch family
   (`inference/patches/sched-latch-fix/`, `inference/patches/hicache-mamba-fix/`)
   following each directory's README (re-anchor, self-check, smoke).
3. On an isolated test instance (same profiles, different container name and host
   port — never against live traffic), run the acceptance gates in
   `inference/tools/acceptance/`; they exit non-zero on failure.
4. Only then switch the profile's `image:` line and update the pinned base digest in the
   installers (`SGLANG_BASE_DIGEST` / `DERIVED_IMAGE` in `scripts/online/setup.sh` and
   `scripts/offline/export.sh`; `expected_image` in `scripts/offline/setup.sh`).
5. Record the migration: add an `evidence/<name>/` bundle, link it from
   `evidence/README.md` (+ `README-zh.md`), and refresh the top-level READMEs (supported
   version, patch retirement table, scheduler anchors).

## Scripts in tools/

Stdlib-only where possible, endpoints and sizes parameterized via env/flags (no
absolute paths, no credentials). The acceptance suites double as documentation of
expected scheduler behavior — new discriminating tests go there.
