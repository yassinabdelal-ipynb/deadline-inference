# Verification status

Version: **0.3.1**. Updated 2026-10-16.

This educational image-inference desktop application requires no account, API
key or paid service. The Windows download runs on CPU; NVIDIA CUDA execution
uses the separately configured source environment.

## Completed locally

- Full hardware-enabled source suite: **96 tests passed, no skips**, in **28.681
  seconds**. Command: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/test.ps1 -Hardware`.
- A fresh native desktop smoke test explicitly requested **CUDA**, completed both
  real inference replays, and verified actual backend identity, results, final
  button state and no web server on the RTX 5060 Laptop GPU.
  Command: `.venv/Scripts/python.exe -m deadline_engine desktop --self-test --smoke-inference --test-device cuda --output results/release-0.3.1-native-cuda.json`.
- Python and PowerShell setup routes refused an unintended replacement of the
  existing CUDA environment with CPU before installing anything.
- Source checks inspect an explicit allowlist for accidental weights,
  environments, credentials and private home paths. This bounded publication
  check is not a security audit.
- The corrected prior measurement snapshot covers four classifiers, eight
  CPU/CUDA profiles, 160 timed forwards and 32 scheduling replays. All **1,824**
  submitted requests have outcomes: **936 on time, 101 late and 787 expired**.
  These are measured experiments, not a guarantee that every request finishes.

## Release verification

The versioned Windows CPU bundle is being rebuilt and checked separately.
Consult the release notes and `docs/public/verification.json` for the final
extracted-application result. Source CUDA detection never substitutes for
running the frozen CPU application's own libraries.

## Compatibility boundaries

Local verification used Windows 11 x64, Python 3.14.7, torch 2.12.0+cu130,
torchvision 0.27.0+cu130, driver 596.08 and an RTX 5060 Laptop GPU with 8,151 MiB
VRAM. The frozen build uses separate CPU-only dependencies.

GitHub workflows provide Windows/Linux/macOS reference and real CPU-model
checks. Their definition is not a passing result: consult GitHub Actions.
No physical second PC, Linux desktop, Mac, other NVIDIA GPU, code signing or
production-scale deployment is certified by these local results.

See [testing](docs/TESTING.md), [portability](docs/PORTABILITY.md) and
[measurements](docs/public/README.md) for commands and limitations.
