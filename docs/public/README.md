# Measurement snapshot

These are observations from a bounded experiment on **one interactive Windows
laptop**, not a hardware score, accuracy test or promise of a scheduling win.
The compact files contain only selected measurements and checkpoint identities;
they omit local paths, raw diagnostic output and bulk per-request traces.

| File | Contents |
| --- | --- |
| [measurements.json](measurements.json) | Hardware/version scope, input settings, four checkpoint URLs/SHA256 values, 160 timing samples, 32 replay summaries. |
| [profiles.csv](profiles.csv) | 16 model/device/batch rows; mean, sample SD, forward time, host-service rate and memory. |
| [replays.csv](replays.csv) | One row per scheduling replay, retaining policy, seed index, schedule hash, counts, latency percentiles and generator lateness. |
| [throughput.svg](throughput.svg) | Images per total measured host-service second, batch 1/4, CPU/CUDA. |
| [outcomes.svg](outcomes.svg) | Combined accounting by model/device; not a policy ranking. |

## Conditions

- Windows 11 x64 build 26200, Python 3.14.7, PyTorch 2.12.0+cu130,
  torchvision 0.27.0+cu130; NVIDIA driver 596.08.
- RTX 5060 Laptop GPU, 8,151 MiB VRAM. CPU reports
  `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel`, 16 logical processors.
  CPU measurements use two PyTorch threads through the source CUDA-library
  installation, not the frozen CPU executable.
- Procedural normalized RGB float32 images, 224 × 224, 1,000 output classes.
  Predictions do not evaluate accuracy. Four official checkpoints were fetched
  explicitly and validated through the local state-dictionary import function.
- Eight model/device profiles. Each measures 10 forwards at batch 1 and 10 at
  batch 4: **160 timed forwards**, after preparation/calibration and two
  additional warmups per measured batch size.
- Scheduling uses a 0.6-second bursty arrival window, base rate 40/s,
  multiplier 5, deadline 1,000 ms, queue 128, collection 4 ms and batch limit 4.
  Seeds 7/8 alternate policy order. Four models × two devices × two seeds ×
  two policies = **32 replays /16 matched schedule pairs**.

This is a short compatibility/measurement fixture, not a sustained overload
capacity study. Rates and tails vary with laptop power, temperature, other
processes and OS scheduling. No experiment here isolates their contributions.

## What the numbers mean

Host service time includes input assembly, transfer, forward computation,
prediction return and synchronization. Profile rate is **total images divided
by summed host service time**, not the average of individual sample rates.
CUDA forward time uses synchronized events; CPU forward time uses a host clock.
Those narrower timings are retained separately.

For example, SqueezeNet1.1 CUDA batch 4 observed **177.67 images/s**;
its host-service mean was **22.51 ±2.95 ms** per batch, with 10 samples.
EfficientNet-B0 CPU batch 4 observed **1,105.95 ±782.17 ms**. This high sample
SD is why the chart should not be read as a universal ranking. ResNet18 CPU
batch 4 observed **4.08 images/s**, compared with **4.21** at batch 1.

Scheduling throughput includes queue drain and excludes setup/warmup. End-to-end
latency starts at planned arrival, including generator delay. p50/p95/p99 include
**completed requests only**, with their sample counts reported. Deadline success
is on-time completions divided by **all submitted requests**, so expiry and
rejection cannot manufacture a successful score.

Across all replays:

```text
1,824 submitted = 936 completed on time + 101 completed late + 787 expired
```

There were zero rejected, failed, cancelled-queued or cancelled-in-flight
requests in this fixture. An expired request produced no answer. Each seed has
57 planned requests per replay; comparison pairs retain identical schedule
hashes. Counts pooled across policies show accounting coverage, not superiority.
Use the separate policy rows in the CSV before making a comparison.

All generated requests have the same relative deadline and priority. EDF
therefore follows arrival order here; the policy comparison also changes
batching and collection. It cannot isolate an EDF ordering advantage.

CUDA peak allocated memory measures PyTorch tensor allocations, not all GPU
memory. Reserved memory includes allocator cache. Process memory is a lifetime
high-water mark including preparation and prior work. These scopes do not
support isolated per-model total-memory rankings or electrical-power claims.

## Arrival-clock correction

Independent analysis found that a Windows event wait could return slightly
before its requested arrival. The generator now rechecks the monotonic clock in
a loop, and a regression fixture deliberately wakes early. The **32 scheduling
replays above were rerun after this fix**; only the unaffected device profiles
were reused with matching model hashes and configuration. Preliminary scheduling
numbers are excluded from this public snapshot. Exact conservation alone would
not have detected an early-submission error.

## Reproduce or redraw

After source setup and explicit model downloads:

```powershell
.\.venv\Scripts\python.exe scripts/validate_models.py --fetch --devices cpu cuda --repeats 2 --output results/image-models
```

The command runs real models and creates local raw evidence. It is bounded but
can take several minutes; use only devices actually available. The validation
script records per-model/device analysis alongside each report.

Redraw checked-in figures without installing PyTorch or downloading weights:

```sh
python scripts/render_results.py --from-public docs/public/measurements.json
```

To export a new matrix, pass its matching environment report. Review conditions
and labels before replacing this snapshot:

```powershell
.\.venv\Scripts\python.exe -m deadline_engine doctor --output results/my-environment.json
.\.venv\Scripts\python.exe scripts/render_results.py --source results/image-models/validation.json --environment results/my-environment.json
```

The renderer uses Python's standard library. Its allowlist retains numerical
results and identities without copying personal paths from diagnostics. Commit
the compact public outputs after inspection, rather than model weights,
environments or full traces. Repeated runs will produce different timings.
