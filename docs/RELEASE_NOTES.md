# Release notes

## 0.3.1

A native desktop release for image-classification execution and scheduling
experiments. The Windows x64 downloadable application bundles CPU inference;
NVIDIA CUDA uses the pinned source installation. No account, API key, password,
browser server or paid service is required to use either route.

Included capabilities:

- Four model families: MobileNetV3-Small, ResNet18, EfficientNet-B0 and
  SqueezeNet1.1. Explicit hash-checked pretrained downloads and trusted compatible
  local `.pth`/`.pt` state-dictionary imports.
- Direct device profiling at batch 1 and the selected batch limit, with repeated
  timing samples, throughput, parameters and scoped memory reporting.
- Seeded steady/bursty/overload traffic, bounded queues, singleton FIFO and
  deadline-aware compatible batching. Sequential matched schedule replay.
- Native white desktop controls, cancellation, live counts, latency plots and
  JSON/CSV export; no browser or JavaScript framework.
- Device choices follow actual runtime capability. A CPU-only installation
  cannot gain CUDA support by selecting a device name.
- A monotonic arrival-wait correction prevents early submissions after an early
  event wakeup. The corrected 32-replay measurement matrix accounts for all
  1,824 submissions; preliminary scheduling figures are not public benchmarks.
- Portable writable-cache handling, explicit setup/diagnostic/test commands,
  pinned dependencies, license notices and downloadable checksum manifests.

### Verification scope

The local source suite passed **96 tests with no skips in 28.681 seconds**,
including real CPU/CUDA model and release-infrastructure checks. A fresh native
desktop CUDA smoke test passed on the actual RTX 5060. These are local results;
they do not establish identical performance or desktop compatibility elsewhere.

The checked-in [measurement snapshot](public/README.md) records real CPU and
CUDA execution on one Windows 11 RTX 5060 laptop, four imported official
checkpoints, eight profiles and 32 corrected scheduling replays. It is not a
measurement from every supported computer or a claim that all requests completed.

Release verification details and the exact artifact checksums accompany the
[GitHub release](https://github.com/yassinabdelal-ipynb/deadline-inference/releases/tag/v0.3.1).
Do not interpret a prepared CI workflow as a passing remote run before GitHub
has executed it.

### Known limits

- Windows application packages are CPU-only, x64 and unsigned. Extract the whole
  archive; do not distribute the executable alone. Cached model inference works
  offline, but the explicit first package/model download needs internet access.
- Only four image-classification architectures and float32 execution are
  supported. Procedural predictions do not evaluate model accuracy.
- Linux/macOS source routes are provided; separate physical desktop verification
  remains pending. No ROCm, DirectML, MPS or Windows ARM binary support is claimed.
- Equal generated relative deadlines make EDF order match FIFO. Batching and
  collection also differ, so this is not an isolated EDF ordering comparison.
- The runtime estimate is heuristic. Laptop power/thermal/OS activity can change
  timing; there is no hard real-time or production-server performance guarantee.

Project code uses the MIT license. Libraries and checkpoints retain their own
licenses and conditions. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
