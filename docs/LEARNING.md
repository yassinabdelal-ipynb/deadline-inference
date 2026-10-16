# Learning guide

This project tests how supported image classifiers run on a computer, and how
organizing their incoming work changes response times. The Python scheduler,
arrival generator, model registry/import checks, accounting, exports, tests and
desktop interface are the educational implementation. PyTorch supplies tensor
operations and device execution; torchvision supplies the four supported
network structures and published ImageNet weights. NVIDIA's driver and
PyTorch's CUDA runtime execute GPU operations. The project does not train a
model, invent a neural network architecture or provide custom CUDA kernels.

Synthetic random RGB inputs are useful because both policies can receive exactly
the same inputs without private data. Their class predictions do not measure
accuracy on photographs or any real application.

## Start with the purpose

Imagine many people asking a photo app to analyze their pictures. Each picture
is a request. The queue is their waiting line. A scheduler decides who runs
next. FIFO means first in, first out: the earliest arrival runs first. In this
app its baseline handles one picture per model call. A batch puts several
compatible pictures into one call. A deadline is the latest useful answer time.

The app offers two experiments. **Device test** asks, "How quickly does this
image model execute on this CPU or GPU?" **Run**, using Compare, asks, "When
requests arrive over time, which policy returns more answers before deadlines?"
The second adds queueing and overload to the first. A larger batch can improve
images per second yet increase waiting, so neither policy is promised to win.

## Model structure, weights and classes

The architecture is the arrangement of model layers. Its weights are the
learned numbers inside those layers. A state dictionary stores named parameter
and buffer tensors, without supplying architecture code. The app knows four
structures: MobileNetV3-Small, ResNet18, EfficientNet-B0 and SqueezeNet1.1.
Imported weights must match the selected structure and output class count.

A class is one possible output category. Official checkpoints have 1,000. A
custom model trained for ten categories needs a matching ten-class checkpoint;
typing 10 into the app does not train it. All supported inputs have three RGB
channels and the chosen square size. See [MODELS.md](MODELS.md) for import steps.

Use your own or trusted checkpoints. The tensor-only loader and strict matching
are format checks, not permission to execute files from unknown sources. The
app never relaxes loading to execute checkpoint Python code or substitutes
random weights for missing ones.

## Follow one request

1. `make_schedule` creates a fixed list of IDs, arrival offsets, deadlines, and
   input indices from a configuration and seed. This happens before replay.
2. The generator waits for each planned arrival using a monotonic clock. It
   attempts submission even if earlier requests have not completed. A late
   generator records its lateness instead of moving the planned arrival forward.
3. A submission that is already past its planned deadline expires immediately.
   Otherwise a bounded queue admits it if a waiting slot is available. A full
   queue rejects a still-live submission immediately. In-flight work is separate
   from the waiting queue.
4. The dedicated inference worker selects work according to the policy. FIFO
   takes one request at a time. Deadline-aware scheduling chooses urgent work
   and can combine requests only when their shape and dtype match.
5. A request whose deadline has already passed before inference expires. A
   started request finishes as on time or late according to its absolute
   deadline. Cancellation and inference failures have separate terminal states.
6. The native Tk window consumes snapshots. Tk widgets stay on the main thread; model work and rendering have separate ownership.

An immutable request is a record whose fields cannot be reassigned. Its ID stays
attached to the same planned work through admission, batching, inference, and
export. That makes conservation checks useful: a request cannot disappear just
because the system is overloaded.

## Why batches help, and why waiting can hurt

A GPU can often process several inputs together more efficiently than separate
one-input launches. Larger batches can also use more memory and make an urgent
request wait for less urgent companions. The deadline policy limits both batch
size and collection time.

The backend measures service time after warmup for each supported batch size.
Its runtime estimate is the slowest of three calibration samples multiplied by
1.25, with another 1 ms added. The scheduler uses that estimate to stop collecting
before it would knowingly use up an urgent request's remaining time. It is a
heuristic: runtime and Windows thread scheduling vary, so meeting an estimate
does not guarantee meeting a deadline. The FIFO baseline uses batch size one.

Priority is stored in request metadata. Do not claim priority fairness or
starvation prevention unless a policy actually uses it and those properties
have been measured.

## Reading the results honestly

- **Submitted:** arrivals whose submission was attempted. Future arrivals stopped
  by cancellation are counted separately as unsubmitted cancelled work.
- **Admitted:** submitted requests accepted into the queue.
- **Rejected:** submitted requests denied because the queue was full.
- **Expired:** requests discarded before inference after their deadline. This
  includes a generator-late submission that expires before queue admission.
- **Completed on time / completed late:** inference returned a prediction before
  or after the absolute deadline.
- **Cancelled queued / cancelled in flight:** admitted work interrupted by the
  run's cancellation state. In-flight GPU operations are allowed to return
  safely; the result is still counted as cancelled.
- **Failed:** submitted work for which execution failed. Read the exported error.
- **Throughput:** completed requests divided by measured replay wall time. It is
  useful alongside the on-time count, because late answers may be stale.
- **Deadline success:** completed on time divided by all submitted requests.
  Rejecting work cannot improve that fraction by hiding it from the denominator.
- **p50/p95/p99 latency:** percentiles over completed requests' end-to-end
  latencies. Rejected and expired requests are not completions; read their
  counts beside the percentiles. Very small samples make tail percentiles weak
  evidence.
- **Generator lateness:** actual submission time minus planned arrival time.
  End-to-end latency starts at planned arrival, so a stalled generator cannot
  make the service appear faster by resetting the clock.

CUDA events time the model's device forward pass. Host service timing includes
input assembly, transfers, inference, prediction transfer, and synchronization.
End-to-end latency also includes generator lateness and queue waiting. CUDA
operations are asynchronous: a CPU timer surrounding an unsynchronized launch
would mostly time submission, which is why this backend synchronizes correctly.

Peak allocated and reserved CUDA memory have different meanings. Allocated is
live tensor storage; reserved includes the caching allocator's held blocks. The
process memory high-water mark includes preparation and earlier work in the
same process and is not an isolated per-policy memory measurement.

## What the device test adds

After preparation and warmup, the device test measures batch 1 and the selected
maximum batch repeatedly. A sample is one timed batch execution. Mean is the
average; standard deviation describes how much those samples varied. The
report preserves individual samples so a surprising average can be investigated.
Its images/second uses the total processed images divided by summed measured
host service time. It excludes a traffic queue and is distinct from replay
throughput, which includes the arrival window and its final drain.

Parameter count is the number of learned values. Parameter bytes count tensor
storage for those parameters, not every temporary tensor or the full process.
Neither predicts accuracy. The profile's CUDA memory peaks reset after each
batch's warmup; the process high-water mark retains earlier allocations.
These measurements describe the selected workload, not universal "GPU power",
FLOPS, electrical consumption or hardware health.

The traffic comparison also reports mean executed batch size, forward time per
image and p95 queue wait. Mean batch size tells you whether a limit of eight
actually produced groups of eight. Queue wait measures time admitted work spent
waiting before execution; it differs from full latency, which begins at the
planned arrival. Read each report's sample count and scope beside a number.

## Five interview questions you can answer from this code

**1. Why use an open-loop generator?**

Arrivals are planned independently of responses. A closed-loop client that
waits for each response reduces its offered load when the server slows, which
can hide overload. Here both policies replay the same schedule and input
indices. Compare schedule hashes and generator lateness in the exports.

**2. How do you know requests are not lost?**

Each submitted ID has exactly one terminal outcome: completed on time, completed
late, rejected, expired, cancelled queued, cancelled in flight, or failed. Tests
sum those categories back to submitted and check IDs are unique. Future
cancelled arrivals appear explicitly but are excluded from submitted. Queue
bounds are checked separately because conservation alone does not prevent
unbounded memory growth.

**3. Why have one inference worker?**

It gives one owner to model execution. The generator can keep offering work and
the local UI can read progress without mutating the model concurrently. The
backend also rejects overlapping calls. A lock protects shared queue/accounting
state, and cancellation wakes waiting work. Python threads do not imply parallel
Python computation; the value here is coordinating waiting, I/O, and PyTorch's
native device operations.

**4. Why might deadline-aware batching lose to FIFO?**

Collection delay, a poor runtime estimate, small offered load, CPU overhead, or
already-efficient one-input inference can outweigh a batch's device efficiency.
Under overload, more expiry can reduce completed latency while discarding more
work. Compare all terminal counts, deadline success across all submissions, and
repeated runs before describing a policy as better.

**5. What did you implement versus reuse?**

This repository implements the two policies, bounded queue, deterministic load
schedule, explicit request lifecycle, measurements, exports, and local UI. It
reuses Python's threading and native Tk facilities, PyTorch inference and CUDA events,
and torchvision's image architectures and official weights. This is an educational service,
not evidence of production-scale performance or research novelty. Describe your
own contributions to the code accurately, including use of coding assistance.

## Exercises

1. Run the same seed and rate with batch limits 1, 2, 4, and 8. Predict which
   setting should win before reading results. Compare completed latency and
   deadline success, then explain any disagreement with your prediction.
2. Hold the schedule fixed and shorten the deadline. Find the first run with
   expiry. Explain why completed p95 can improve while fewer arrivals succeed.
3. Reduce queue capacity under overload. Record rejection, expiry, completed
   work, and peak queue depth. Explain which resource the bound protects.
4. Inspect the calibrated estimates. Modify the safety multiplier in a local
   branch, then repeat the same workload. Show whether underestimation creates
   more late completions; do not infer this from a single run.
5. Use Device test to compare the four official models at the same image and
   batch sizes on CPU. Repeat on CUDA if available. Explain why a ranking can
   change between devices without asserting that accuracy changed.
6. Import your own compatible trained checkpoint. Deliberately select the wrong
   family or class count and explain why validation rejects it. Do not relabel
   a randomly initialized fixture as a trained model.
7. Add a test with incompatible input metadata. Then design a third policy
   using priority and measure waiting time per priority class. Include a
   low-priority starvation case before claiming the extension is fair.

## Resume wording

Template with blank metrics:

> Implemented a Python/PyTorch single-GPU inference scheduler with bounded
> queues, deadline-aware batching, reproducible open-loop replay, and request
> conservation tests; measured ___ requests/s and ___% on-time completion at
> ___ offered requests/s on ___ across ___ repeated runs.

Fill blanks only from saved benchmark files. Name the device and workload, keep
the baseline comparison if claiming an improvement, and avoid implying model
training, production deployment, or a new scheduling algorithm. The README's
observed results state which hardware paths were actually verified.
