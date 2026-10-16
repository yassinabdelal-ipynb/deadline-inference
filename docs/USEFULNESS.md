# What this tool is useful for

This application lets someone answer a concrete question on their own computer:
**when image requests arrive in bursts, how does batching affect timely results,
waiting time, and discarded work?** It replays the same planned arrivals through
two scheduling policies and records what actually happened on the selected
CPU or NVIDIA GPU.

It also measures a selected image classifier directly, without arrival queues:
repeated batch timing, images/second, parameter count and available memory
metrics. Users can choose four supported image classification families and
import compatible trained weights. This broadens the experiments while keeping
the input/output contract understandable. See [MODELS.md](MODELS.md).

## Who benefits

| User | Useful task | Evidence the app produces |
| --- | --- | --- |
| Student learning systems and ML | Understand queues, batching, deadlines and overload with real inference | Live queue/batch activity, separate terminal outcomes, reproducible exports |
| Developer exploring a local image-classification workload | Try compatible custom weights, measure CPU/CUDA batches, then test a scheduling hypothesis | Checkpoint identity, repeated device timings, memory scope, matched schedules and deadline-success measurements |
| Instructor or workshop host | Demonstrate why throughput and freshness differ | Repeatable steady/bursty/overload inputs; CPU path for learners without an NVIDIA GPU |
| Portfolio reviewer | Inspect engineering discipline beyond a static screenshot | Independent scheduler core, cancellation, conservation tests, real model checks, explicit measurement limitations |

For example, a learner can keep the seed and arrival pattern fixed, change the
batch limit from one to four, and observe whether more requests finish before
their deadline. Next they can tighten the deadline and see why waiting to fill
a batch stops helping. The exported report lets another person inspect the
exact inputs, device and outcomes behind the conclusion.

## What its measurements can establish

The app can compare these two policies for this model, input shape, device,
configuration and run. It can expose queue growth, overload rejection, request
expiry, completed-late work, generator stalls, and cancellation. Repeated runs
show variability rather than hiding it behind a single number. Identical seeds
give identical planned schedules; operating-system and GPU timing still vary.

This is a focused educational lab. It is not a general PC optimizer, a GPU
health/stability test, a neural-network accuracy evaluator, or a production model
server. Running it does not accelerate other programs. Random image classes have
no meaningful accuracy score. Its runtime estimate is a heuristic, so it cannot
guarantee hard real-time deadlines. Each experiment uses one selected supported
classifier, one fixed RGB image shape and one inference device. Import support
does not cover arbitrary model code, detectors, segmentation or generation.
Production servers solve additional problems
such as multiple models/users, isolation, networking, dynamic shapes, model
lifecycle and security.

A desktop executable makes the experiment easier to open; portability does not
make benchmark results transferable between computers. A slower deadline policy
is still a valid observation. A policy that lowers measured latency by rejecting
more work has not automatically become more useful. Always report deadline
success as a fraction of **all submitted** requests alongside discarded work.

## An honest portfolio description

"Implemented a local inference scheduling laboratory with bounded queues,
deadline-aware batching, open-loop traffic replay, a native desktop interface,
and per-request accounting. Compared FIFO and batching on [device] over [number]
replays, reporting [measured result] with [measured variability/limitation]."

Fill the brackets only from your own saved results. Credit PyTorch and
torchvision for the neural network and execution kernels. Explain the
algorithms and experiments you can demonstrate in the code. Do not claim a new
scheduling algorithm or a performance improvement on workloads you did not test.

If people need a compact way to learn or demonstrate these tradeoffs, this tool
has a clear purpose. Wider adoption would need real feedback from those users;
this project does not claim established demand, production readiness, or a
performance advantage over production inference systems.
