# Changelog

## 0.3.1 — 2026-10-16

- Prepare the first public source and Windows CPU desktop release.
- Restrict the desktop device selector to detected capabilities and explain CPU-only runtimes.
- Reject unsupported CUDA selections before starting an experiment.
- Identify the version and CPU/CUDA runtime in the window title.
- Extend native smoke verification to explicitly request and check CUDA execution.
- Separate portable CPU downloads from the optional NVIDIA CUDA source setup.
- Add measured charts, reproducible evidence, repository templates and release checks.

The image-model expansion supports four classifier families, compatible local
tensor checkpoint imports, repeated device profiles and matched scheduling
replays. Measurements and their limits are described in the README. Earlier
local versions were development builds; this log does not imply earlier public releases.
