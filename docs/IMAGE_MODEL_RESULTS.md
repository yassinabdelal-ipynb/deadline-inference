# Image-model expansion: actual local results

This is a bounded compatibility and measurement check on the development
Windows laptop. Four real public pretrained checkpoints were fetched explicitly,
copied through the same local import function used by the desktop, and run on
both CPU and CUDA. Procedural inputs do not evaluate prediction accuracy.
These observations do not establish a general model ranking or hardware score.

## What ran

- Command: `.venv\Scripts\python.exe scripts/validate_models.py --fetch --devices cpu cuda --repeats 2 --output results/image-models`.
- Host: Windows 11 x64, Python 3.14.7, RTX 5060 Laptop GPU with 8,151 MiB VRAM,
  driver 596.08. Source libraries: torch 2.12.0+cu130 and torchvision 0.27.0+cu130.
  CPU inventory exposes `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel`
  and 16 logical processors; no marketing CPU model name is inferred.
  CPU measurements use CPU execution through that source library installation,
  not the separately frozen CPU-only executable.
- Each model/device profile: float32 RGB 224x224 inputs; batch 1 and batch 4;
  10 timed samples per batch, 20 per profile, 160 measured forwards in total.
  CPU uses two PyTorch threads. Model loading/calibration and two additional
  warmups per measured batch size are excluded from these timed samples.
- Scheduling: 0.6-second bursty windows, base rate 40/s, burst multiplier 5,
  deadline 1,000 ms, batch limit 4, queue 128, collection 4 ms, seeds 7 and 8.
  FIFO/deadline order alternates between seeds. Four models x two devices x
  two seeds x two policies gives 32 replays /16 matched schedule pairs.
- Evidence: `results/image-models/validation.json`, the per-model/device
  `profile/profile.json` and CSV, `comparison/comparison.json` and CSV, and
  `results/image-model-validation-output.txt`.

## Device profiles

Host service includes assembly, transfer, model forward, prediction return and
synchronization. Values below are mean +/- sample standard deviation in ms for
one batch. Images/s is total measured images divided by summed service time;
it is not the arithmetic mean of per-sample rates. Each table cell has 10
samples behind its batch measurement.

| Model | Device | Batch 1 service, ms | Batch 4 service, ms | Batch 1 images/s | Batch 4 images/s |
| --- | --- | ---: | ---: | ---: | ---: |
| MobileNetV3-Small | CPU | 67.91 +/- 2.82 | 150.40 +/- 18.57 | 14.73 | 26.60 |
| MobileNetV3-Small | CUDA | 51.35 +/- 9.01 | 68.62 +/- 45.83 | 19.48 | 58.29 |
| ResNet18 | CPU | 237.28 +/- 12.24 | 980.18 +/- 189.46 | 4.21 | 4.08 |
| ResNet18 | CUDA | 50.55 +/- 13.98 | 109.28 +/- 48.05 | 19.78 | 36.60 |
| EfficientNet-B0 | CPU | 287.88 +/- 160.76 | 1105.95 +/- 782.17 | 3.47 | 3.62 |
| EfficientNet-B0 | CUDA | 68.35 +/- 6.26 | 100.32 +/- 38.68 | 14.63 | 39.87 |
| SqueezeNet1.1 | CPU | 149.77 +/- 46.97 | 423.79 +/- 45.46 | 6.68 | 9.44 |
| SqueezeNet1.1 | CUDA | 19.06 +/- 1.88 | 22.51 +/- 2.95 | 52.47 | 177.67 |

Variation is substantial in several rows. For example, EfficientNet-B0 CPU
batch 4 has service-time SD of 782.17 ms against a mean of 1,105.95 ms.
This was an interactive Windows session, not an isolated sustained benchmark.
The measurements do not isolate the cause of the variation. Retest under
controlled conditions before making a capacity, speedup or model-ranking claim.
The ResNet18 CPU batch-4 result also illustrates that batching need not improve
images/s. The app reports that result rather than assuming batching wins.

Forward-only means and SD, raw samples and percentiles are retained in each
profile. On CUDA those use synchronized CUDA events; CPU forward time uses a
host clock. They have a narrower scope than the service times above.

## Model size and memory

| Model | Parameters | Parameter storage, MiB | CUDA batch-4 peak allocated, MiB |
| --- | ---: | ---: | ---: |
| MobileNetV3-Small | 2,542,856 | 9.70 | 51.76 |
| ResNet18 | 11,689,512 | 44.59 | 255.66 |
| EfficientNet-B0 | 5,288,548 | 20.17 | 126.52 |
| SqueezeNet1.1 | 1,235,496 | 4.71 | 95.07 |

Parameter storage excludes buffers and temporary tensors. Allocated CUDA memory
is PyTorch tensor storage; reserved memory additionally includes allocator cache.
Peaks reset after the measured batch size's warmup, but held cache can include
earlier work in the same process. The OS process peak is a lifetime high-water
mark that includes imports, preparation and previous runs. It reached
1410.68 MiB in this matrix. Driver/context/display memory is not fully captured.
These scopes do not support ranking isolated per-model total memory consumption.

## Scheduling accounting

The first matrix exposed a generator timing bug during independent analysis:
Windows Event.wait could return slightly before its planned monotonic arrival.
The generator now rechecks the clock in a loop, and an EarlyEvent regression
fixture deliberately wakes early. The 32 scheduling replays were rerun after
fixing this. Their corrected totals below replace the preliminary run.
The eight device profiles have no arrival generator; they were reused with
exact configuration, device and checkpoint-hash checks.

Actual scheduling rerun command:

```powershell
.\.venv\Scripts\python.exe scripts/validate_models.py --devices cpu cuda --repeats 2 --reuse-profiles --output results/image-models
```

| Model | Device | Submitted | On time | Late | Expired |
| --- | --- | ---: | ---: | ---: | ---: |
| MobileNetV3-Small | CPU | 228 | 67 | 23 | 138 |
| MobileNetV3-Small | CUDA | 228 | 175 | 8 | 45 |
| ResNet18 | CPU | 228 | 11 | 12 | 205 |
| ResNet18 | CUDA | 228 | 190 | 5 | 33 |
| EfficientNet-B0 | CPU | 228 | 114 | 11 | 103 |
| EfficientNet-B0 | CUDA | 228 | 144 | 10 | 74 |
| SqueezeNet1.1 | CPU | 228 | 28 | 29 | 171 |
| SqueezeNet1.1 | CUDA | 228 | 207 | 3 | 18 |

The table combines both policies and both seeds per model/device to show
accounting coverage; it is not a policy comparison. Inspect the separate replay
rows for throughput, latency percentiles/sample counts, mean batch size, queue
wait, deadline success and discarded work before drawing a policy conclusion.

Across corrected replays: **1,824 submitted = 936 completed on time +
101 completed late + 787 expired**. There were no rejected, failed,
cancelled-queued or cancelled-in-flight requests, and no unsubmitted cancelled
arrivals. The integration script passed its paired schedule/input, shape/batch
bound, queue and terminal-count checks. Independent analysis then passed all
eight corrected reports: 32 replays /16 pairs, reconstructing schedule hashes,
IDs, batch membership, terminal counts, throughput and latency percentiles. It
also checks that each submission timestamp is at or after its planned arrival.
Evidence is in each comparison/analysis.json and results/image-model-analysis.json.
The first integration PASS alone did not detect early wakeup.

A large expiry count is an observed outcome, not success for all arrivals.
These short busy fixtures verify imported model execution and request
accounting. The static runtime estimate and machine timing can produce late
completions or expiry even with a one-second deadline. No universal performance
advantage is claimed.

## What this establishes

The four official network checkpoints passed local import and real CPU/CUDA
execution through the new multi-model path. It exercises the actual import
function; native file-dialog selection is an additional UI workflow check.
It does not prove compatibility with arbitrary models, malicious files, every
custom training checkpoint or another computer. Custom-class fixtures in the
tests verify architecture handling; randomly initialized fixtures are not
presented as trained models or benchmarked accuracy.

The older MobileNet-only CPU/CUDA measurements in RESULTS.md and
REVISION_CPU_RESULTS.md remain historical. They use different workloads/session
conditions and predate the corrected monotonic arrival wait. They are retained
as historical records, not new v0.3 timing-validity evidence. Do not merge them
with this matrix or use them as the expanded-release benchmark.


## Extracted v0.3 Windows CPU executable

The complete new archive was extracted into a fresh local path containing
spaces. Its own executable ran four real image-model smoke cases, reporting
PyTorch 2.12.0+cpu, torchvision 0.27.0+cpu and native Tk 9.0.4. All checks passed,
including final Run enabled /Stop disabled and no listening server.

| Model | FIFO completed /expired | Deadline completed /expired |
| --- | ---: | ---: |
| MobileNetV3-Small | 12 /0 | 12 /0 |
| ResNet18 | 5 /7 | 7 /5 |
| EfficientNet-B0 | 12 /0 | 12 /0 |
| SqueezeNet1.1 | 12 /0 | 12 /0 |

Each policy submitted 12 requests. Across all eight replays, 96 submitted =84
completed +12 expired, with zero failed work. The pass checks functional model
execution, accounting and native state, not a guarantee that every request
completes or meets its deadline. Do not infer all completions were on time from
this combined completion count.

MobileNet used its explicit built-in checkpoint cache; the other three cases
provided official downloaded checkpoint files as local imported state-dictionary
paths. Their backend SHA256 values matched the registry. This exercises the
frozen application's own libraries and model path rather than borrowing the
source interpreter. It does not claim a manual file-picker import completed:
the native picker opened, and its import callbacks/backend were tested
separately. This is one host's local verification, not a second-PC test.

Actual command: `.build-venv\Scripts\python.exe scripts/verify_release.py
--smoke --all-models --output results/image-model-release-verification.json`.
Evidence: that JSON and `results/image-model-release-output.txt`. Models are
cached/downloaded separately and remain excluded from the distributable.
