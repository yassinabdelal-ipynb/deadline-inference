# Running on another computer

Deadline Inference is a native desktop application. It requires no account,
password, API key, browser, paid service, Node, Docker or WSL. It uses local
compute. Package installation and explicit model downloads need an internet
connection; cached models and imported checkpoints can run offline.

## Choose a download

Get the files from [GitHub Releases](https://github.com/yassinabdelal-ipynb/deadline-inference/releases).

| Computer | Route | Verification boundary |
| --- | --- | --- |
| Windows x64, CPU | `DeadlineInference-Windows-x64-CPU.zip` | Frozen application and source exercised on the development Windows 11 laptop; another physical PC remains unverified |
| Windows x64, NVIDIA CUDA | Source ZIP and CUDA setup below | Real pretrained inference tested on an RTX 5060 Laptop GPU; other GPU/driver combinations require a local check |
| Linux x64 desktop, CPU or compatible NVIDIA CUDA | Source setup | Automated CI targets Linux; physical desktop verification remains separate |
| macOS desktop, Intel or Apple Silicon | Source setup, CPU | Automated CI targets macOS CPU; physical Mac verification remains separate |
| AMD/Intel GPU or Apple Silicon GPU | Use CPU execution | ROCm, DirectML and MPS acceleration are not implemented |
| Windows ARM, 32-bit OS, headless desktop | No packaged desktop support claim | CLI/reference tests may be evaluated independently |

There is no universal performance or compatibility guarantee. Supported package
wheels, driver compatibility, RAM, free disk space and a graphical desktop still
matter. Check the actual Actions result for a particular commit; a workflow
file alone does not mean its jobs passed. Hosted CI has no CUDA validation.

## Windows CPU app: no Python installation

1. Download the CPU ZIP and extract the **entire** folder to a local directory.
2. Open `DeadlineInference/DeadlineInference.exe`. Keep `_internal` and the
   license files beside it; copying the executable alone does not work.
3. Select a classifier and click **Fetch model**. This explicitly downloads its
   public pretrained checkpoint, verifies its SHA-256 hash and caches it.
4. Run **Device test** for direct inference measurements or **Run** for traffic
   replay. Synthetic inputs do not evaluate prediction accuracy.

This executable contains CPU-only PyTorch. It does not offer CUDA. Selecting a
different device cannot add missing GPU libraries; use the source route for
NVIDIA execution. The portable download includes Python, Tk and required CPU
libraries. It is unsigned; an unknown-publisher prompt is possible. The archive
is a portable folder, not an installed Windows service or a store application.

Check download integrity with PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath '.\DeadlineInference-Windows-x64-CPU.zip'
```

Compare the result with `SHA256SUMS.txt` from the same release. A checksum detects
corruption; it is not a code signature or independent security audit.

## Windows source: CPU or NVIDIA CUDA

Install official **64-bit Python 3.14** with pip and Tcl/Tk from
[python.org](https://www.python.org/downloads/windows/) if needed. Extract the
source ZIP to a writable directory. No compiler, system CUDA toolkit or shell
activation is required for the pinned prebuilt packages.

```powershell
Set-Location -LiteralPath 'C:\Projects\Deadline Inference'
# CPU installation:
py -3.14 scripts/bootstrap.py --backend cpu --download-weights
# OR NVIDIA installation instead:
py -3.14 scripts/bootstrap.py --backend cuda --download-weights
.\.venv\Scripts\python.exe -m deadline_engine doctor
.\.venv\Scripts\python.exe -m deadline_engine desktop
```

Choose one backend during initial setup. CUDA requires a supported NVIDIA GPU
and compatible installed driver. Setup never changes drivers, enables WSL,
installs system compilers or reboots. If the device is unavailable, the app
explains the problem instead of relabeling CPU measurements as CUDA.

Setup is repeatable with the same backend. It **refuses to replace an existing
CPU environment with CUDA, or CUDA with CPU**, unless `--replace-backend` is
explicit. Keep two installations instead when useful:

```powershell
py -3.14 scripts/bootstrap.py --backend cpu --environment '.venv-cpu'
py -3.14 scripts/bootstrap.py --backend cuda --environment '.venv-cuda'
.\.venv-cuda\Scripts\python.exe -m deadline_engine desktop
```

The PowerShell wrapper supports `-Backend`, `-DownloadWeights`, `-Python`,
`-Environment` and `-ReplaceBackend`. The direct Python command avoids any need
to change PowerShell execution policy. **Run Deadline Inference.cmd** starts the
default `.venv`; a custom environment needs its own interpreter command above.

## Linux and macOS source

Use Python 3.14 with pip, venv and Tcl/Tk. A graphical desktop session is required
for Tk. Some Linux distributors provide Tk separately, such as `python3-tk`;
install the package matching your Python through the distributor's instructions.
Setup leaves this system package choice to you.

```sh
python3.14 -m tkinter
python3.14 scripts/bootstrap.py --backend cpu --download-weights
./.venv/bin/python -m deadline_engine doctor
./.venv/bin/python -m deadline_engine desktop
```

Linux x64 NVIDIA users can select `--backend cuda` with a compatible driver.
macOS uses CPU/PyPI wheels. The current release pins Python 3.14, torch 2.12.0
and torchvision 0.27.0; CUDA uses the official cu130 index. If no matching wheel
exists for your platform, keep the error and hardware details rather than
removing pins at random. GUI availability and wheel compatibility are separate.

## Data and model boundaries

The Windows bundle caches data under `%LOCALAPPDATA%\DeadlineInference` and
does not require its extracted folder to be writable. Source checkouts use
`.cache`. `DEADLINE_ENGINE_DATA_DIR` explicitly overrides the data location.
Exports go to the path you choose. No telemetry or remote inference is required.

Weights are excluded from release archives. The four built-in classifiers use
hash-pinned official checkpoints; fetching is explicit and cached. Imports are
trusted `.pth`/`.pt` state dictionaries for the supported architectures only.
They are not a general model format or a sandbox for untrusted files. See
[MODELS.md](MODELS.md) and [SECURITY.md](../SECURITY.md).

## Build and verify a release

On Windows, use a **separate CPU build environment**:

```powershell
py -3.14 scripts/bootstrap.py --backend cpu --environment '.build-venv'
.\.build-venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.build-venv\Scripts\python.exe scripts/check_release.py --source
.\.build-venv\Scripts\python.exe scripts/build_release.py
.\.build-venv\Scripts\python.exe scripts/check_release.py --archives
```

Outputs are `dist/DeadlineInference-Windows-x64-CPU.zip`,
`dist/deadline-inference-source.zip` and `dist/SHA256SUMS.txt`. Intermediate
application files live in `dist/v<package-version>/DeadlineInference`; previous
version directories are preserved. `--skip-build` refreshes documentation and
ZIPs from that version's existing executable. `--source-only` produces only
source and its checksum, on any supported Python host.

The build records exact dependencies and gathers Python, Tcl/Tk and dependency
licenses. The publication checker rejects archive traversal, unapproved source
files, environments, weights, common credential signatures and personal home
paths. It does not establish that arbitrary code or third-party binaries are
safe. Raw machine diagnostics and large request traces stay out of source ZIPs.

The release workflow builds and checks the Windows CPU package, explicitly
downloads the default checkpoint for an extracted-app smoke test, and retains
the verified archives. Manual dispatch builds artifacts without publishing.
Pushing a matching `v<version>` tag publishes assets only after those checks
pass. GitHub supplies the short-lived `github.token`; maintainers do not add
personal keys or secrets. Only the publishing job has `contents: write`.

Actions are pinned to verified upstream release commits:
[checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1),
[setup-python v7.0.0](https://github.com/actions/setup-python/releases/tag/v7.0.0),
[upload-artifact v7.0.0](https://github.com/actions/upload-artifact/releases/tag/v7.0.0).
Token permissions follow [GitHub's documentation](https://docs.github.com/en/actions/tutorials/authenticate-with-github_token).
Tk requirements follow [Python's documentation](https://docs.python.org/3.14/library/tkinter.html).
