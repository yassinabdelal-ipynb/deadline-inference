# Testing

Test the scheduler, real inference, native window and packaged application
separately. A passing reference test does not prove that CUDA works. Record the
OS, application version, selected device, model/checkpoint hash and commands run.
See [STATUS.md](../STATUS.md) for the latest recorded checks and
[measured results](IMAGE_MODEL_RESULTS.md) for the published experiment.

## Clean source installation

Use a new writable folder, preferably with a space in its name. Do not copy
another computer's virtual environment. Install 64-bit Python 3.14 with Tcl/Tk
as described in [PORTABILITY.md](PORTABILITY.md), then run from the project root:

```powershell
py -3.14 scripts/bootstrap.py --backend cpu --download-weights
.\.venv\Scripts\python.exe -m deadline_engine doctor
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m deadline_engine desktop
```

On Linux/macOS, use `python3.14` for bootstrap and `./.venv/bin/python` for the
project interpreter. A graphical desktop and matching Tcl/Tk are required for
the window. Reference tests work without inference dependencies:

```sh
python3.14 -m unittest discover -s tests -v
```

Real-model tests are opt-in; display-dependent tests skip explicitly when Tk
or a display is unavailable. Check reported skips before describing coverage.
Downloading packages/checkpoints needs network access; cached inference needs
no account, API key or network service.

## Real CPU and CUDA checks

After explicitly fetching the default MobileNet weights, Windows CPU checks are:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test.ps1 -Model
.\.venv\Scripts\python.exe -m deadline_engine desktop --self-test --smoke-inference --test-device cpu --output results/native-cpu.json
```

On a compatible NVIDIA computer, explicitly replace the CPU test environment
with the CUDA dependency set, then verify real execution. Close the app first.
Setup does not install or change a GPU driver:

```powershell
py -3.14 scripts/bootstrap.py --backend cuda --replace-backend --download-weights
.\.venv\Scripts\python.exe -m deadline_engine doctor
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test.ps1 -Hardware
.\.venv\Scripts\python.exe -m deadline_engine desktop --self-test --smoke-inference --test-device cuda --output results/native-cuda.json
```

Bootstrap reuses `.venv` and preserves an existing CPU/CUDA installation by
default. A different backend is refused unless `--replace-backend` is supplied.
For two independent installations, use `--environment` with another directory
and run that environment's interpreter. Do not run setup while an experiment
is using the environment being changed. On a POSIX shell, the explicit test
gates are:

```sh
RUN_MODEL_TESTS=1 RUN_GPU_TESTS=1 ./.venv/bin/python -m unittest discover -s tests -v
```

Omit `RUN_GPU_TESTS=1` for CPU-only testing. An explicitly requested unavailable
GPU must fail rather than silently use CPU. For the native smoke report, check
`checks_passed: true`, `inference_tested: true`, `status: completed`,
`server_started: false`, two policy summaries and the requested device in
`model_backends`. A CUDA report must identify the real GPU and contain positive
device timing and CUDA memory measurements. `torch.cuda.is_available()` or
`nvidia-smi` alone is insufficient proof of inference.

`--self-test` without `--smoke-inference` checks startup/layout only. The smoke
opens real widgets, runs a short comparison and closes automatically. Manual
keyboard, file-dialog and resizing checks are still necessary.

## Desktop acceptance

1. Choose MobileNetV3-Small, Official pretrained, 1,000 classes, size 224 and
   CPU. Explicitly select **Fetch model** if its weights are missing.
2. Select compare, steady traffic, 20 requests/s, 1 second, 1,000 ms deadline,
   batch limit 2, queue 32, collection wait 4 ms and seed 7. Run it. Loading and
   warmup precede measurement. Both policy columns must show actual results.
3. Save JSON and CSV to a writable folder. Check the same `schedule_hash` for
   paired policies, actual backend/model identity and request accounting.
4. Run a 15-second fixture, then press Stop/Escape while work is active. The
   window must remain responsive and the report must account for cancellation.
   An issued inference call returns safely before its worker stops.
5. Try an invalid rate or batch size. Expect a useful validation error and no
   background run. Correct it, then run again.
6. Use Tab, Ctrl+Enter, Escape and Ctrl+S. Resize panes and scroll the inputs,
   results and log. Export is enabled only when a report exists. System check
   should identify the actual runtime. A CPU-only executable must offer CPU
   alone and explain how CUDA source setup differs.
7. Close while idle and during a run. Workers must finish cancellation before
   shutdown. Reopen and repeat a cached run offline.

For every policy, verify this exact identity:

```text
submitted = completed_on_time + completed_late + rejected + expired
          + cancelled_queued + cancelled_in_flight + failed
```

`admitted` is intermediate, not another terminal category. Future arrivals
stopped before submission are `cancelled_before_arrival`, counted separately as
`unsubmitted_cancelled`. Expiry/rejection on a slow computer is a measured
workload outcome, not by itself a failed application test. Normal completed
fixtures must have no inference errors or cancelled work.

## Model import and device profiling

Repeat the short fixture and **Device test** with each supported family:
MobileNetV3-Small, ResNet18, EfficientNet-B0 and SqueezeNet1.1. Fetch each missing
official checkpoint explicitly. Device test should report the actual checkpoint
hash, batch sizes, sample counts, mean/SD, images/s and memory scope. It measures
model execution, not accuracy, electrical power or general GPU health.

For an import, select the matching architecture and import its cached official
`.pth` file using **Import weights...**. Check the copied checkpoint's SHA256
against the original, then run Device test and Compare. Changing architectures
must clear a previously selected import. A checkpoint selected under the wrong
family or class count must fail with a clear error and no random fallback.
Use only trusted files; tensor-only loading is not a security sandbox.

For a CLI profile or the bounded four-model compatibility matrix:

```powershell
.\.venv\Scripts\python.exe -m deadline_engine fetch-weights --model resnet18
.\.venv\Scripts\python.exe -m deadline_engine profile --device cpu --model resnet18 --batch 2 --samples 10 --output results/resnet18-profile
.\.venv\Scripts\python.exe scripts/validate_models.py --fetch --devices cpu cuda --repeats 2 --output results/model-validation
```

Use `--devices cpu` without CUDA. Omit `--fetch` once all four checkpoints are
cached. The matrix uses two seeds, batch 1/4 profiles with 10 samples each and
sequential policy replays. For both profile and replay, test cancellation during
preparation and active work; zero-sample cancelled reports must retain their
configuration and explicit cancellation state. Format/dtype/finite-value import
cases are also covered by automated tests; see [MODELS.md](MODELS.md).

## Repeat a measurement

```powershell
.\.venv\Scripts\python.exe -m deadline_engine benchmark --device cpu --duration 1 --rate 40 --batch 4 --repeats 3 --output results/my-benchmark
.\.venv\Scripts\python.exe scripts/analyze_results.py results/my-benchmark/comparison.json
```

This benchmark gives 18 replays: three traffic patterns, three repeats and two
policies. Choose CUDA only after the real-device checks. Keep model, image size,
seed and settings fixed for comparisons; record power mode and competing work.
Inspect throughput, all-submission deadline success, completed-request latency,
discarded work, generator lateness and memory together. No speedup is required
to pass. Warmup is excluded and repeat variability remains part of the result.

## Packaged Windows application

Download the Windows x64 CPU ZIP from GitHub Releases, verify SHA256SUMS.txt and
extract the **entire** archive. Keep `_internal` and `THIRD_PARTY_LICENSES` with
`DeadlineInference.exe`. Test on another Windows x64 PC without activating or
copying a development environment. CPU use needs neither Python nor NVIDIA
hardware. The executable is unsigned; do not automatically bypass OS warnings.

Open it, explicitly fetch weights, then repeat the desktop checks above. After
the default checkpoint is cached, this PowerShell check waits for the actual
windowed process and handles paths with spaces. Run beside the extracted
`DeadlineInference` directory:

```powershell
$appExecutable = (Resolve-Path -LiteralPath '.\DeadlineInference\DeadlineInference.exe').Path
$appReport = Join-Path $env:TEMP 'deadline-inference-self-test.json'
$appArguments = @('--self-test', '--smoke-inference', '--test-device', 'cpu', '--output', ('"{0}"' -f $appReport))
$appProcess = Start-Process -FilePath $appExecutable -ArgumentList $appArguments -WindowStyle Hidden -Wait -PassThru
if ($appProcess.ExitCode -ne 0) { throw "Application smoke test failed: $($appProcess.ExitCode)" }
Get-Content -LiteralPath $appReport
```

The report must identify CPU libraries from the packaged runtime. Selecting
CUDA cannot add CUDA libraries to a CPU-only binary. Check that a cached second
run works offline and that cache writes go to the user's writable app-data
directory. Review personal paths before sharing diagnostic reports.

Maintainers can check archive contents and exercise an extracted release with:

```powershell
.\.build-venv\Scripts\python.exe scripts/check_release.py --source --archives
.\.build-venv\Scripts\python.exe scripts/verify_release.py --smoke --all-models --output results/release-verification.json
```

The four-model check needs all checkpoints already cached; it downloads none.
It must test the extracted executable's own libraries. A workflow passing on
hosted Windows/Linux/macOS runners provides additional evidence, but does not
certify every interactive desktop or a different physical NVIDIA GPU. Record
second-PC/OS checks only after someone actually performs them.
