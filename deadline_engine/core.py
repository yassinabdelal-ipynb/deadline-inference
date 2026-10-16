"""Bounded, open-loop scheduling. No PyTorch or web dependencies."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
import threading
import time
from .errors import PreparationCancelled

TERMINAL = ("completed_on_time", "completed_late", "rejected", "expired",
            "cancelled_queued", "cancelled_in_flight", "failed")


@dataclass(frozen=True)
class Config:
    arrival_rate: float = 100.0
    burst_intensity: float = 5.0
    deadline_ms: float = 100.0
    max_batch: int = 8
    policy: str = "compare"
    seed: int = 7
    pattern: str = "bursty"
    duration_s: float = 4.0
    queue_capacity: int = 128
    collection_ms: float = 4.0
    device: str = "cuda"
    model_id: str = "mobilenet_v3_small"
    weights_path: str = ""
    num_classes: int = 1000
    image_size: int = 224

    def validate(self):
        ranges = {"arrival_rate": (1, 2000), "burst_intensity": (1, 10),
                  "deadline_ms": (1, 5000), "duration_s": (0.05, 30),
                  "collection_ms": (0, 100), "max_batch": (1, 16),
                  "queue_capacity": (1, 2048), "seed": (0, 2**32 - 1),
                  "num_classes": (1, 10000), "image_size": (64, 512)}
        for key, (lo, hi) in ranges.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{key} must be a finite number in [{lo}, {hi}].")
        for key in ("max_batch", "queue_capacity", "seed", "num_classes", "image_size"):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f"{key} must be an integer.")
        for key, choices in {"policy": ("compare", "fifo", "deadline"),
                             "pattern": ("steady", "bursty", "overload"),
                             "device": ("cpu", "cuda")}.items():
            if getattr(self, key) not in choices:
                raise ValueError(f"{key} must be one of {choices}.")
        from .models import get_model_spec
        get_model_spec(self.model_id)
        if not isinstance(self.weights_path, str) or len(self.weights_path) > 32767:
            raise ValueError("weights_path must be a file path string.")
        if not self.weights_path and self.num_classes != 1000:
            raise ValueError("Official pretrained weights have 1000 classes. Import matching weights for a different class count.")
        return self


@dataclass(frozen=True)
class PlannedRequest:
    id: int
    offset_s: float
    deadline_ms: float
    priority: int
    input_index: int
    shape: tuple = (3, 224, 224)
    dtype: str = "float32"


@dataclass(frozen=True)
class Request:
    id: int
    arrival_s: float
    deadline_s: float
    priority: int
    input_index: int
    shape: tuple
    dtype: str


def make_schedule(config: Config) -> list[PlannedRequest]:
    """Piecewise Poisson arrivals, planned before any model execution."""
    config.validate()
    rng = random.Random(config.seed)
    schedule = []
    # 100 ms rate segments. Bursts occur in first 200 ms of each second.
    segments = math.ceil(config.duration_s / 0.1)
    for segment in range(segments):
        start = segment * 0.1
        end = min(start + 0.1, config.duration_s)
        multiplier = 1.0
        if config.pattern == "bursty":
            multiplier = config.burst_intensity if segment % 10 < 2 else 0.25
        elif config.pattern == "overload" and start >= config.duration_s / 3:
            multiplier = config.burst_intensity
        t = start + rng.expovariate(config.arrival_rate * multiplier)
        while t < end:
            schedule.append(PlannedRequest(len(schedule), t, config.deadline_ms, 0, rng.randrange(32),
                                           shape=(3, config.image_size, config.image_size)))
            if len(schedule) > 10000:
                raise ValueError("Schedule exceeds 10,000 requests; reduce rate, duration, or burst intensity.")
            t += rng.expovariate(config.arrival_rate * multiplier)
    return schedule


def schedule_hash(schedule):
    return hashlib.sha256(json.dumps([asdict(x) for x in schedule], sort_keys=True).encode()).hexdigest()


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def select_batch(queue, policy, max_batch):
    """Pure batch selection: FIFO singleton, or earliest deadline then priority/id."""
    if not queue:
        return []
    if policy == "fifo":
        return queue[:1]
    ordered = sorted(queue, key=lambda r: (r.deadline_s, -r.priority, r.id))
    first = ordered[0]
    return [r for r in ordered if (r.shape, r.dtype) == (first.shape, first.dtype)][:max_batch]


def collection_wait_s(batch, queue, now, config, backend, first_seen):
    if config.policy == "fifo" or len(batch) >= config.max_batch or not batch:
        return 0.0
    # Reserve enough time for ANY possible compatible final batch size.
    estimate = max(backend.estimate_s(n) for n in range(len(batch), config.max_batch + 1))
    slack = min(r.deadline_s for r in queue) - now - estimate
    return max(0.0, min(first_seen + config.collection_ms / 1000 - now, slack))


def replay(config, schedule, policy, backend, cancel=None, on_update=None):
    """One generator + one inference worker. Clock origin is AFTER warmup.

    Submitted means the generator reached admission. Deadlines are anchored to
    PLANNED monotonic arrival, so generator stalls cannot conceal latency.
    """
    config.validate()
    if policy not in ("fifo", "deadline"):
        raise ValueError("Replay policy must be fifo or deadline.")
    # Validate the entire trace before starting a worker. Besides clearer errors,
    # this prevents malformed metadata from stranding already-admitted work.
    for plan in schedule:
        if not isinstance(plan, PlannedRequest):
            raise ValueError("Schedule entries must be PlannedRequest instances.")
        for name in ("id", "input_index"):
            value = getattr(plan, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Schedule {name} must be a nonnegative integer.")
        for name in ("offset_s", "deadline_ms"):
            value = getattr(plan, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (name == "deadline_ms" and value == 0)):
                raise ValueError(f"Schedule {name} must be finite and {'positive' if name == 'deadline_ms' else 'nonnegative'}.")
        if isinstance(plan.priority, bool) or not isinstance(plan.priority, int):
            raise ValueError("Schedule priority must be an integer.")
        if (not isinstance(plan.shape, tuple) or not plan.shape
                or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in plan.shape)):
            raise ValueError("Schedule shape must be a nonempty tuple of positive integers.")
        if not isinstance(plan.dtype, str) or not plan.dtype.strip():
            raise ValueError("Schedule dtype must be a nonempty string.")
    if len({p.id for p in schedule}) != len(schedule):
        raise ValueError("Schedule IDs must be unique.")
    if any(a.offset_s > b.offset_s for a, b in zip(schedule, schedule[1:])):
        raise ValueError("Schedule must be ordered by arrival.")
    cancel = cancel if cancel is not None else threading.Event()
    cond = threading.Condition()
    ready = threading.Event()
    queue = []
    rows = {}
    timeline = []
    batches = []
    live_trace = deque(maxlen=256)
    state = {"submitted": 0, "admitted": 0, "queue_depth": 0, "in_flight": 0,
             "peak_queue_depth": 0, **dict.fromkeys(TERMINAL, 0)}
    origin = [0.0]
    producer_done = [False]
    errors = []
    observer_errors = []

    def snapshot(force=False):
        now = time.perf_counter()
        state["queue_depth"] = len(queue)
        state["peak_queue_depth"] = max(state["peak_queue_depth"], len(queue))
        state["cancelled"] = state["cancelled_queued"] + state["cancelled_in_flight"]
        if force or not timeline or now - origin[0] - timeline[-1]["t_s"] >= 0.05:
            item = {"t_s": max(0, now - origin[0]), **state}
            timeline.append(item)
            if on_update and not observer_errors:
                try:
                    on_update({**state, "trace": list(live_trace)})
                except Exception as exc:
                    # A display failure must not change request outcomes or leave
                    # pending work behind. Disable the broken observer and report
                    # the error separately from inference failures.
                    observer_errors.append(f"Progress callback failed: {exc}")

    def finish(req, status, now, **extra):
        row = rows[req.id]
        if row["status"] != "pending":
            raise RuntimeError(f"Duplicate terminal outcome for {req.id}")
        row.update(status=status, finish_s=now - origin[0],
                   latency_ms=max(0, (now - req.arrival_s) * 1000), **extra)
        state[status] += 1
        if status in ("completed_on_time", "completed_late"):
            live_trace.append({"id": req.id, "status": status, "latency_ms": row["latency_ms"]})

    def worker():
        try:
            if hasattr(backend, "set_cancel_event"):
                backend.set_cancel_event(cancel)
            backend.prepare()
        except PreparationCancelled:
            ready.set()
            return
        except Exception as exc:
            errors.append(f"Backend preparation failed: {exc}")
            ready.set()
            return
        ready.set()
        with cond:
            cond.wait_for(lambda: origin[0] > 0)
        try:
            while True:
                with cond:
                    now = time.perf_counter()
                    if cancel.is_set():
                        for req in queue:
                            finish(req, "cancelled_queued", now)
                        queue.clear()
                        snapshot(True)
                        return
                    for req in queue[:]:
                        if req.deadline_s <= now:
                            queue.remove(req)
                            finish(req, "expired", now)
                    if not queue:
                        snapshot()
                        if producer_done[0]:
                            return
                        cond.wait(0.01)
                        continue
                    first_seen = min(rows[r.id]["submission_s"] + origin[0] for r in queue)
                    while policy == "deadline":
                        batch = select_batch(queue, policy, config.max_batch)
                        now = time.perf_counter()
                        delay = collection_wait_s(batch, queue, now, config, backend, first_seen)
                        if delay <= 0 or cancel.is_set() or producer_done[0]:
                            break
                        cond.wait(min(delay, 0.01))
                        if any(r.deadline_s <= time.perf_counter() for r in queue):
                            break
                    if cancel.is_set():
                        continue
                    now = time.perf_counter()
                    # Recheck after collection waiting; never send already stale input.
                    for req in queue[:]:
                        if req.deadline_s <= now:
                            queue.remove(req)
                            finish(req, "expired", now)
                    batch = select_batch(queue, policy, config.max_batch)
                    if not batch:
                        continue
                    for req in batch:
                        queue.remove(req)
                    state["in_flight"] = len(batch)
                    batch_id = len(batches)
                    snapshot()
                started = time.perf_counter()
                try:
                    measured = backend.infer(batch)
                    if len(measured["predictions"]) != len(batch):
                        raise RuntimeError("Backend prediction count does not match batch size.")
                    for key in ("device_ms", "service_ms"):
                        if (isinstance(measured[key], bool) or not isinstance(measured[key], (int, float))
                                or not math.isfinite(measured[key]) or measured[key] < 0):
                            raise RuntimeError(f"Backend {key} must be finite and nonnegative.")
                except Exception as exc:
                    with cond:
                        now = time.perf_counter()
                        batches.append({"batch_id": batch_id, "size": len(batch),
                                        "start_s": started - origin[0], "finish_s": now - origin[0],
                                        "status": "failed", "error": str(exc),
                                        "device_ms": None, "service_ms": (now - started) * 1000})
                        for req in batch:
                            finish(req, "failed", now, error=str(exc), executed=True, batch_id=batch_id)
                        raise
                ended = time.perf_counter()
                with cond:
                    batches.append({"batch_id": batch_id, "size": len(batch),
                                    "start_s": started - origin[0], "finish_s": ended - origin[0],
                                    "status": "completed",
                                    "device_ms": measured["device_ms"], "service_ms": measured["service_ms"]})
                    for req, prediction in zip(batch, measured["predictions"]):
                        status = "cancelled_in_flight" if cancel.is_set() else (
                            "completed_on_time" if ended <= req.deadline_s else "completed_late")
                        finish(req, status, ended, executed=True, prediction=prediction, batch_id=batch_id)
                    state["in_flight"] = 0
                    snapshot()
        except Exception as exc:
            with cond:
                errors.append(f"Inference worker failed: {exc}")
                now = time.perf_counter()
                for req in queue:
                    finish(req, "failed", now, error=str(exc))
                queue.clear()
                state["in_flight"] = 0
                cancel.set()
                cond.notify_all()
                snapshot(True)

    thread = threading.Thread(target=worker, name="inference-worker", daemon=False)
    thread.start()
    ready.wait()
    if errors:
        thread.join()
        raise RuntimeError(errors[0])
    with cond:
        origin[0] = time.perf_counter() + 0.02
        cond.notify_all()
    try:
        for plan in schedule:
            arrival = origin[0] + plan.offset_s
            # A timed wait may return slightly early on Windows. Recheck the
            # monotonic target rather than admitting work before its planned
            # arrival (which could produce negative relative submission times).
            while not cancel.is_set():
                remaining = arrival - time.perf_counter()
                if remaining <= 0 or cancel.wait(remaining):
                    break
            if cancel.is_set():
                break
            req = Request(plan.id, arrival, arrival + plan.deadline_ms / 1000,
                          plan.priority, plan.input_index, plan.shape, plan.dtype)
            with cond:
                # Worker may have failed between the wait and acquiring the lock.
                if cancel.is_set():
                    break
                now = time.perf_counter()
                rows[req.id] = {"id": req.id, "status": "pending", "submitted": True,
                    "admitted": False, "arrival_s": plan.offset_s, "deadline_s": req.deadline_s-origin[0],
                    "submission_s": now-origin[0], "generator_lateness_ms": max(0, (now-arrival)*1000),
                    "input_index": req.input_index, "shape": req.shape, "dtype": req.dtype,
                    "priority": req.priority, "executed": False, "finish_s": None, "latency_ms": None}
                state["submitted"] += 1
                if now >= req.deadline_s:
                    finish(req, "expired", now)
                elif len(queue) >= config.queue_capacity:
                    finish(req, "rejected", now)
                else:
                    queue.append(req)
                    rows[req.id]["admitted"] = True
                    state["admitted"] += 1
                snapshot()
                cond.notify_all()
    finally:
        with cond:
            producer_done[0] = True
            cond.notify_all()
        thread.join()
    ended = time.perf_counter()
    with cond:
        snapshot(True)
    for plan in schedule:
        if plan.id not in rows:
            rows[plan.id] = {"id": plan.id, "status": "cancelled_before_arrival", "submitted": False,
                            "admitted": False, "arrival_s": plan.offset_s, "input_index": plan.input_index,
                            "deadline_s": plan.offset_s + plan.deadline_ms / 1000,
                            "priority": plan.priority, "shape": plan.shape, "dtype": plan.dtype,
                            "executed": False, "latency_ms": None, "finish_s": None}
    trace = [rows[p.id] for p in schedule]
    counts = Counter(r["status"] for r in trace if r["submitted"])
    if any(r["status"] == "pending" for r in trace) or sum(counts.values()) != state["submitted"]:
        raise RuntimeError("Request conservation failed.")
    completed = counts["completed_on_time"] + counts["completed_late"]
    successful_batches = [b for b in batches if b["status"] == "completed"]
    executed_images = sum(b["size"] for b in successful_batches)
    forward_times = [b["device_ms"] for b in successful_batches]
    batch_by_id = {b["batch_id"]: b for b in batches}
    queue_waits = [max(0, (batch_by_id[r["batch_id"]]["start_s"] - r["submission_s"]) * 1000)
                   for r in trace if r.get("executed") and r.get("batch_id") in batch_by_id]
    latencies = [r["latency_ms"] for r in trace if r["status"].startswith("completed_")]
    lateness = [r["generator_lateness_ms"] for r in trace if r["submitted"]]
    elapsed = max(config.duration_s if not cancel.is_set() else 0, ended-origin[0], 1e-9)
    submitted = state["submitted"]
    summary = {**state, "planned": len(schedule), "unsubmitted_cancelled": len(schedule)-submitted,
        "completed": completed, "elapsed_s": elapsed, "throughput_rps": completed/elapsed,
        "completion_rate": completed/submitted if submitted else 0,
        "deadline_success_fraction": counts["completed_on_time"]/submitted if submitted else 0,
        "deadline_misses": counts["expired"] + counts["completed_late"],
        "discarded": counts["rejected"] + counts["expired"],
        "latency_ms": {name: percentile(latencies, q) for name,q in (("p50",.5),("p95",.95),("p99",.99))},
        "latency_sample_count": len(latencies),
        "queue_wait_ms": {"p50": percentile(queue_waits, .5), "p95": percentile(queue_waits, .95)},
        "mean_batch_size": executed_images / len(successful_batches) if successful_batches else None,
        "forward_ms": {"p50": percentile(forward_times, .5), "p95": percentile(forward_times, .95)},
        "forward_ms_per_image": sum(forward_times) / executed_images if executed_images else None,
        "generator_lateness_ms": {"p95": percentile(lateness,.95), "max": max(lateness,default=0)},
        "peak_memory": backend.peak_memory(), "batch_count": len(batches),
        "failed_batch_count": sum(b["status"] == "failed" for b in batches),
        "device_timing_complete": all(b["device_ms"] is not None for b in batches),
        "device_ms_total": sum(b["device_ms"] for b in batches if b["device_ms"] is not None),
        "service_ms_total": sum(b["service_ms"] for b in batches)}
    return {"policy": policy, "config": asdict(config), "schedule_hash": schedule_hash(schedule),
            "backend": backend.metadata(), "summary": summary, "trace": trace,
            "timeline": timeline, "batches": batches, "errors": errors, "observer_errors": observer_errors}
