# Security and privacy

The desktop application runs locally. It has no account system, API keys,
telemetry, cloud inference requirement, or listening web server. Upload means
importing a file into the local cache; it does not transmit the file. Procedural
inputs contain no user photographs. The app does not modify GPU drivers, install
compilers, enable WSL, or permanently change execution policy.

## Model downloads and imports

Setup obtains dependencies from the configured package indexes. The explicit
**Fetch model** action retrieves only the selected official torchvision
checkpoint. The app checks its pinned identity and records the file SHA256.
Model preparation never downloads missing weights or silently falls back to
random weights. Once dependencies and the selected checkpoint are cached,
experiments work offline.

Import only weights you made yourself or obtained from a source you trust.
Imports accept tensor state dictionaries for the selected supported image
classification family. Loading uses `weights_only=True`, strict key/shape/dtype
matching, and no custom Python modules, unsafe-global allowlisting, or fallback
to unrestricted pickle loading. A mismatch is an error, not a reason to relax
those checks.

The supported path uses float32 learned parameters and the architecture's
expected integer buffers. It rejects half-precision, float64, complex,
quantized/sparse and non-finite checkpoint tensors rather than guessing a
conversion. Finite values are checked before and after model loading. PyTorch's
legacy compatibility for an omitted BatchNorm training-step counter is retained;
missing learned parameters are never substituted. File/archive/tensor limits
bound ordinary input sizes but cannot make an untrusted file safe.

These restrictions reduce risk but **are not a security sandbox**. Malformed
files can still exhaust resources or encounter native-library vulnerabilities.
PyTorch documents the limits of tensor-only loading in its
[serialization security guidance](https://docs.pytorch.org/docs/2.12/notes/serialization.html#weights-only-security).
Do not import untrusted files merely because they end in `.pth`/`.pt` or have
a checksum. A checksum identifies bytes; it does not certify safe behavior,
authorship, training quality, licensing, or provenance.

This app does not accept arbitrary executable model programs, ONNX/TensorRT
engines, or plugins from a checkpoint. Extending the supported architectures
requires reviewed application source changes. See [model format requirements](docs/MODELS.md).

## Data locations and exports

Source checkouts use `.cache`; the frozen Windows app stores cached files in
`%LOCALAPPDATA%\DeadlineInference\cache`. `DEADLINE_ENGINE_DATA_DIR` can override
the writable data root. Avoid a shared writable directory unless you trust its
other users. Importing a checkpoint makes a cached copy; removing the original
does not remove that copy.

Exports and system diagnostics may contain hardware details, Python locations,
absolute model/cache paths and original checkpoint names. Review personal paths
before sharing. Source/application archives exclude weights, environments,
caches, raw diagnostics and large request traces. Removing a local export does
not revoke a copy already shared with someone else.

## Reporting a problem

Report ordinary bugs using the repository's bug template with a minimal
reproduction and redacted diagnostics. For a security issue, use
[private vulnerability reporting](https://github.com/yassinabdelal-ipynb/deadline-inference/security/advisories/new)
when available. If GitHub does not offer a private reporting form, open an issue
asking for a private contact without including vulnerability details. Do not
post credentials, exploit files or private checkpoints in an issue.

No security audit, code signing, or universal operating-system certification is
claimed. The local Windows executable is unsigned. Use a source you trust and
compare the archive with its accompanying checksum manifest when available.
