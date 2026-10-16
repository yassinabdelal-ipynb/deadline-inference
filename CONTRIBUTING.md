# Contributing

This is an educational inference scheduling laboratory. Small, reproducible
improvements are welcome. The implementation demonstrates established ideas;
changes must not imply research novelty or guaranteed performance gains.

Read [the README](README.md), [portability](docs/PORTABILITY.md), and
[the learning guide](docs/LEARNING.md) before changing the engine. Keep scheduling
in `deadline_engine/core.py` independent of the desktop interface. Use the
project environment and the pinned dependencies; avoid installing packages into
your system Python.

## Before submitting a change

1. Run `python -m unittest discover -s tests -v` from the source folder. These
   bounded reference tests do not require PyTorch, downloads, or a GPU. Run the
   command using your project interpreter if dependencies are installed there.
2. Add a meaningful test for a corrected scheduler, accounting, export, or
   cancellation defect. Every submitted request must have exactly one terminal
   outcome. Do not replace real-device verification with the fake backend.
3. If you change the UI, follow the manual acceptance checks in
   [TESTING.md](docs/TESTING.md). Record your OS and backend. Keep work off the
   Tk event thread and update widgets only from that thread.
4. If you change inference, run the explicit real-model tests and a short
   comparison on the affected backend. State unavailable hardware plainly.
   A new model family needs an explicit architecture/weight registry entry,
   strict tensor-only checkpoint validation, provenance/license references and
   real import/forward/profile/replay checks. Do not enable execution of Python
   model code from an uploaded file. Keep the image-classification contract and
   unsupported formats clear in the interface and docs.
5. Explain the trigger, resulting behavior, and commands actually run in the
   pull request. Include uncertainty and regressions in measurement reports.

Do not commit virtual environments, cached weights, installed packages, secrets,
personal diagnostic paths, executable build folders, or large request traces.
Do not publish performance claims using only completed-request latency: account
for rejects, expired work, late results, cancellation, and generator lateness.

Project source contributions are accepted under the MIT license in `LICENSE`.
Keep attribution and license notices when reusing third-party material. Model
weights and bundled dependencies retain their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Release preparation

Use the local release/build scripts described in the README. Inspect the source
archive contents and checksum manifest before sharing. Test an extracted release
from a path containing spaces, with no active virtual environment. A frozen
Windows build is a Windows build; it is not a Linux or macOS executable. Keep
dependency license texts with any binary distribution. No automated publication,
signing credentials are required from contributors. The release workflow uses
GitHub's built-in, short-lived repository token; running the application needs
no account or API key. Published artifacts remain unsigned unless a release
explicitly states otherwise.
