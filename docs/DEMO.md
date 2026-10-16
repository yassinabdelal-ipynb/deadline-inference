# 60–90 second native-app demonstration

Do setup and the explicit model download before presenting. Open
`Run Deadline Inference.cmd` or the extracted Windows CPU executable. Confirm
the actual CPU/CUDA backend in the native status bar. No web server is started.

**0–15 seconds:** “This app tests supported image classification models on a CPU
or NVIDIA GPU. Here are four architectures; I can fetch official weights or
import my own matching checkpoint. These procedural images test execution, not
prediction accuracy.” Show the selected model and actual execution device.

**15–30 seconds:** Show a previously measured Device test, clearly labelled as
recorded if necessary. “Warmup precedes these repeated batch timings. This gives
model-specific speed and memory measurements, not a universal hardware score.”

**30–45 seconds:** Choose Compare, Bursty, seed7, duration2 seconds, rate40/s,
deadline250ms, batch4, queue32 and collection4ms. “Both policies replay the same
planned arrivals and inputs. FIFO uses one image at a time; batching combines
compatible images while limiting collection delay.” Select Run.

**45–60 seconds:** Show live queue/batch counts and event log. “A full queue
rejects work; accepted work may expire before execution, finish on time or finish
late. Setup and warmup are outside the measured replay. The estimate helps decide
when to stop waiting, but cannot guarantee deadlines.”

**60–75 seconds:** Read the actual comparison and chart. “Completed latency alone
can hide discarded requests. I compare completion rate, on-time success across
all submitted requests, and explicit rejects/expiry. This accounting line checks
that no request disappeared.” State only values displayed by this run.

**75–90 seconds:** Save JSON or CSV. “These reports preserve settings, model hash, device,
matched schedule hashes and per-request outcomes. The test suite checks bounds,
cancellation and conservation; separate real-model tests verify inference. The
CPU app lets someone without an NVIDIA GPU reproduce the scheduling experiment.”

If calibration takes longer, label any previously saved real result as recorded.
Do not invent metrics. An optional second run demonstrates Escape cancellation,
including queued/in-flight/future planned cancellations. Explain how your own
learning and coding assistance contributed rather than claiming independent
research novelty or production deployment.
