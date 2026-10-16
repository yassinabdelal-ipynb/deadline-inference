# Project guide

Deadline Inference is a native desktop laboratory for studying how image-model
execution and request handling interact. Its purpose is to make an experiment
inspectable: identical offered traffic, visible discarded work and reproducible
configuration. It is not a general model-hosting server or a GPU hardware score.

## Source map

| File | Responsibility |
| --- | --- |
| `deadline_engine/core.py` | Immutable requests, seeded arrival plans, bounded queues, FIFO/deadline batching, outcomes and cancellation. Standard-library reference path. |
| `deadline_engine/backend.py` | Strict weight loading, inference mode, procedural input pool, calibration and CPU/CUDA timing. |
| `deadline_engine/models.py` | Four explicit architecture/checkpoint definitions and pinned hashes. |
| `deadline_engine/profiling.py` | Repeated direct batch measurements without a simulated arrival queue. |
| `deadline_engine/experiment.py` | Matched sequential replays, summaries and JSON/CSV output. |
| `deadline_engine/controller.py` | Background experiment lifecycle and snapshots. |
| `deadline_engine/desktop.py` | Native Tk controls, actual hardware availability, plots and local export. |
| `deadline_engine/paths.py` | Source and frozen-application cache/data locations. |
| `scripts/bootstrap.py` | Project-local pinned dependency installation. |
| `scripts/validate_models.py` | Bounded four-model CPU/CUDA integration matrix. |
| `scripts/analyze_results.py` | Independent checks of accounting, schedule identity and reported metrics. |
| `scripts/render_results.py` | Allowlisted public measurement snapshot, CSVs and SVG figures. |

## One request's journey

1. A seeded plan assigns an immutable ID, planned monotonic arrival, absolute
   deadline, priority and image metadata. Arrival generation does not wait for
   previous responses.
2. At its planned time, the generator attempts admission. A full queue rejects
   the request. Generator lateness is measured rather than hidden.
3. The dedicated worker expires stale queued work. FIFO chooses one oldest live
   request; deadline batching chooses the earliest deadline and groups compatible
   shapes/dtypes up to the limit.
4. Before waiting to collect more work, the worker reserves a calibrated runtime
   estimate for the largest possible batch. The estimate is the slowest of three
   host-service samples ×1.25 +1 ms, after two warmups per batch size. Collection
   time starts at the oldest admission and does not restart on new arrivals.
5. The model executes in eval/inference mode. CPU and GPU timings use their
   appropriate clocks. One worker owns model execution and mutation.
6. Every submitted request receives exactly one terminal outcome. The desktop
   reads snapshots and plots results outside the performance loop.

The collection guard avoids a wait that the current estimate predicts will
violate the earliest deadline. It cannot predict arbitrary OS/driver stalls and
does not provide a hard real-time guarantee. Queue expiry, late completion and
queue rejection remain distinct events.

## Metric definitions

| Metric | Definition |
| --- | --- |
| Throughput | Completed requests /max(arrival window, final outcome time), including drain. |
| Completion rate | Completed requests /submitted requests. |
| Deadline success | On-time completions /submitted requests. |
| End-to-end latency | Final completion time −planned arrival, for completed requests only. |
| Generator lateness | Actual submission time −planned arrival. |
| Discarded | Rejected +expired. Cancellation and failure are shown separately. |
| p95 | 95th percentile of the stated sample population; sample count matters. |

Submitted work must equal completed-on-time +completed-late +rejected +expired
+cancelled-queued +cancelled-in-flight +failed. Future arrivals stopped before
submission have a separate `submitted=false` category. Admitted is an
intermediate count, not another terminal outcome.

## Library and implementation boundaries

PyTorch provides tensors, inference execution, CPU/CUDA kernels, device events
and memory counters. Torchvision provides model architectures and the official
checkpoint definitions. Tk provides native widgets. Python provides threads,
clocks and file/network primitives.

This repository supplies the queueing policies, deterministic traffic generator,
runtime-estimate guard, request accounting, validation, reporting and desktop
workflows. It neither trains the supplied models nor implements their GPU
kernels. Imported weights must be trusted and match an explicitly supported
architecture; restricted tensor loading is not a sandbox for arbitrary files.

## Useful next experiments

The existing comparison changes several things together. A strong extension
would add **FIFO with batching**, then compare it against deadline batching at
the same batch/collection settings. Add mixed relative deadlines next, checking
per-class outcomes and starvation, so EDF order differs meaningfully from FIFO.

Automated rate/batch sweeps could then estimate the highest offered rate meeting
a stated on-time target. Keep raw sample counts, repeat variability, generator
lateness and discarded work visible. Add features only with an experiment that
can tell whether their effect is useful. No universal speedup is assumed.

See [LEARNING.md](LEARNING.md) for explanations, interview questions, modification
exercises and a résumé template, and [the public evidence](public/README.md) for
the observed measurement boundary.
