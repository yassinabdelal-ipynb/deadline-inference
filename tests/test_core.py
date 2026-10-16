"""Bounded CPU-only scheduler tests. The fake backend does not run a model.

Real model/CUDA checks are explicit CLI runs, not hidden test prerequisites.
Small sleeps stand in for service so the actual producer/worker interaction is
tested; assertions use generous timing margins and do not promise performance.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
import unittest

from deadline_engine.core import (
    Config, PlannedRequest, Request, collection_wait_s, make_schedule,
    percentile, replay, select_batch,
)


TERMINALS = (
    "completed_on_time", "completed_late", "rejected", "expired",
    "cancelled_queued", "cancelled_in_flight", "failed",
)


class FakeBackend:
    """Deterministic test reference, deliberately unrelated to GPU throughput."""

    def __init__(self, service_s=0.005, estimate_s=None, fail=False, gate=None):
        self.service_s = service_s
        self.runtime_estimate_s = service_s if estimate_s is None else estimate_s
        self.fail = fail
        self.gate = gate
        self.started = threading.Event()
        self.calls = []
        self.call_times = []
        self.worker_ids = set()
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def prepare(self):
        return None

    def estimate_s(self, batch_size):
        return self.runtime_estimate_s

    def infer(self, requests):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append(list(requests))
            self.call_times.append(time.perf_counter())
            self.worker_ids.add(threading.get_ident())
        self.started.set()
        try:
            if self.gate is not None and not self.gate.wait(2):
                raise RuntimeError("Test inference gate timed out")
            time.sleep(self.service_s)
            if self.fail:
                raise RuntimeError("deliberate fake-backend failure")
            return {
                "device_ms": self.service_s * 1000,
                "service_ms": self.service_s * 1000,
                "predictions": [request.input_index for request in requests],
            }
        finally:
            with self._lock:
                self.active -= 1

    def metadata(self):
        return {"backend": "test-reference", "device": "cpu", "model": "none"}

    def peak_memory(self):
        return {"cuda_peak_allocated_bytes": None, "process_peak_resident_bytes": None}


def config(**changes):
    values = dict(
        arrival_rate=100.0, burst_intensity=4.0, deadline_ms=300.0,
        max_batch=4, policy="compare", seed=7, pattern="bursty",
        duration_s=0.15, queue_capacity=8, collection_ms=3.0, device="cpu",
    )
    values.update(changes)
    return Config(**values)


def planned(index, offset=0.0, deadline_ms=300.0, **changes):
    values = dict(
        id=index, offset_s=offset, deadline_ms=deadline_ms,
        priority=0, input_index=index, shape=(3, 224, 224), dtype="float32",
    )
    values.update(changes)
    return PlannedRequest(**values)


class SchedulerTests(unittest.TestCase):
    def assertConserved(self, result, schedule, capacity):
        trace = result["trace"]
        summary = result["summary"]
        self.assertEqual(len(trace), len(schedule))
        self.assertEqual({row["id"] for row in trace}, {row.id for row in schedule})
        self.assertEqual(len({row["id"] for row in trace}), len(trace))
        submitted = [row for row in trace if row["submitted"]]
        self.assertEqual(summary["submitted"], len(submitted))
        self.assertEqual(summary["submitted"], sum(summary[key] for key in TERMINALS))
        self.assertEqual(len(trace) - len(submitted), summary["unsubmitted_cancelled"])
        for status in TERMINALS:
            self.assertEqual(summary[status], sum(row["status"] == status for row in submitted))
        self.assertEqual(summary["admitted"], sum(row["admitted"] for row in submitted))
        self.assertLessEqual(summary["peak_queue_depth"], capacity)
        for row in submitted:
            self.assertIn(row["status"], TERMINALS)
            self.assertGreaterEqual(row["generator_lateness_ms"], 0)
            if row["status"] in {"completed_on_time", "completed_late"}:
                self.assertGreaterEqual(row["latency_ms"], 0)
                self.assertIsNotNone(row["prediction"])
                self.assertIsNotNone(row["batch_id"])
            if row["executed"]:
                self.assertIn(row["batch_id"], {batch["batch_id"] for batch in result["batches"]})
        for row in trace:
            if not row["submitted"]:
                self.assertEqual(row["status"], "cancelled_before_arrival")

    def test_config_rejects_nonfinite_and_invalid_values(self):
        bad = [
            {"arrival_rate": 0}, {"arrival_rate": math.nan},
            {"arrival_rate": math.inf}, {"deadline_ms": -1},
            {"deadline_ms": math.nan}, {"collection_ms": math.inf},
            {"duration_s": math.nan}, {"burst_intensity": math.inf},
            {"max_batch": 0}, {"max_batch": True}, {"queue_capacity": 0},
            {"collection_ms": -1}, {"policy": "made-up"},
            {"pattern": "made-up"}, {"device": "made-up"},
        ]
        for changes in bad:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**changes).validate()

    def test_requests_and_config_are_immutable(self):
        item = planned(0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            item.offset_s = 1
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config().seed = 4

    def test_schedule_is_seeded_sorted_and_open_loop(self):
        for pattern in ("steady", "bursty", "overload"):
            settings = config(pattern=pattern, duration_s=0.5, arrival_rate=80)
            first = make_schedule(settings)
            self.assertEqual(first, make_schedule(settings))
            self.assertTrue(first)
            self.assertEqual([row.offset_s for row in first], sorted(row.offset_s for row in first))
            self.assertEqual(len(first), len({row.id for row in first}))
            self.assertTrue(all(0 <= row.offset_s < settings.duration_s for row in first))
            self.assertTrue(all(row.deadline_ms == settings.deadline_ms for row in first))

    def test_fifo_batch_one_conservation_and_dedicated_worker(self):
        settings = config()
        schedule = [planned(i, i * 0.003) for i in range(12)]
        backend = FakeBackend()
        parent_id = threading.get_ident()
        result = replay(settings, schedule, "fifo", backend)
        self.assertConserved(result, schedule, settings.queue_capacity)
        self.assertTrue(backend.calls)
        self.assertTrue(all(len(batch) == 1 for batch in backend.calls))
        self.assertEqual(backend.max_active, 1)
        self.assertEqual(len(backend.worker_ids), 1)
        self.assertNotIn(parent_id, backend.worker_ids)

    def test_same_schedule_both_policies_and_batch_limit(self):
        settings = config(queue_capacity=64)
        schedule = [planned(i, 0.0, deadline_ms=2000) for i in range(24)]
        fifo = replay(settings, schedule, "fifo", FakeBackend())
        backend = FakeBackend()
        deadline = replay(settings, schedule, "deadline", backend)
        for result in (fifo, deadline):
            self.assertConserved(result, schedule, settings.queue_capacity)
        self.assertEqual(fifo["schedule_hash"], deadline["schedule_hash"])
        self.assertTrue(any(len(batch) > 1 for batch in backend.calls))
        self.assertTrue(all(len(batch) <= settings.max_batch for batch in backend.calls))
        self.assertEqual(
            {row["id"]: row["prediction"] for row in fifo["trace"]},
            {row["id"]: row["prediction"] for row in deadline["trace"]},
        )

    def test_bounded_queue_rejects_and_preserves_all_requests(self):
        settings = config(queue_capacity=2)
        schedule = [planned(i, 0.0, 1000) for i in range(80)]
        result = replay(settings, schedule, "fifo", FakeBackend(service_s=0.01))
        self.assertConserved(result, schedule, settings.queue_capacity)
        self.assertGreater(result["summary"]["rejected"], 0)

    def test_late_completion_and_queued_expiry_are_distinct(self):
        settings = config(queue_capacity=32)
        class CrossDeadlineBackend(FakeBackend):
            def infer(self, requests):
                # Synchronize against the actual request deadline, rather than
                # assuming a 50 ms window is enough for a hosted OS to schedule
                # both threads. Keep this one call in flight until it is late.
                self.started_before_deadline = time.perf_counter() < requests[0].deadline_s
                target = max(request.deadline_s for request in requests) + 0.005
                while (remaining := target - time.perf_counter()) > 0:
                    time.sleep(min(remaining, 0.01))
                return super().infer(requests)
        backend = CrossDeadlineBackend(service_s=0)
        schedule = [planned(i, 0.0, 500.0) for i in range(8)]
        result = replay(settings, schedule, "fifo", backend)
        self.assertConserved(result, schedule, settings.queue_capacity)
        self.assertTrue(backend.started_before_deadline)
        self.assertGreater(result["summary"]["completed_late"], 0)
        self.assertGreater(result["summary"]["expired"], 0)
        self.assertEqual(result["summary"]["completed_on_time"], 0)

    def test_generator_stall_keeps_planned_deadline_and_records_lateness(self):
        class DelayedArrival(threading.Event):
            def wait(self, timeout=None):
                cancelled = super().wait(timeout)
                time.sleep(0.03)  # Deliberately stall the open-loop generator.
                return cancelled

        settings = config()
        backend = FakeBackend()
        schedule = [planned(0, deadline_ms=10)]
        result = replay(settings, schedule, "fifo", backend, cancel=DelayedArrival())
        self.assertConserved(result, schedule, settings.queue_capacity)
        row = result["trace"][0]
        self.assertGreaterEqual(row["generator_lateness_ms"], 25)
        self.assertAlmostEqual(row["deadline_s"], 0.01, places=7)
        self.assertEqual(row["status"], "expired")
        self.assertFalse(row["admitted"])
        self.assertFalse(backend.calls)

    def test_incompatible_shapes_and_dtypes_never_share_batch(self):
        settings = config(queue_capacity=64)
        schedule = [
            planned(i, shape=(3, 224 + i % 2, 224), dtype="float32" if i % 3 else "float16")
            for i in range(20)
        ]
        backend = FakeBackend(service_s=0.001)
        result = replay(settings, schedule, "deadline", backend)
        self.assertConserved(result, schedule, settings.queue_capacity)
        for batch in backend.calls:
            self.assertEqual(len({(request.shape, request.dtype) for request in batch}), 1)

    def test_tight_deadline_does_not_wait_for_collection_timeout(self):
        settings = config(collection_ms=100, max_batch=8)
        backend = FakeBackend(service_s=0.001, estimate_s=0.2)
        schedule = [planned(0, deadline_ms=150)]
        result = replay(settings, schedule, "deadline", backend)
        self.assertConserved(result, schedule, settings.queue_capacity)
        self.assertTrue(backend.call_times, "An estimate is a wait guard, not a reason to discard live work")
        first = backend.calls[0][0]
        self.assertLess(backend.call_times[0] - first.arrival_s, 0.12)

    def test_wait_guard_with_no_slack_is_exactly_zero(self):
        settings = config(collection_ms=100, max_batch=8)
        backend = FakeBackend(estimate_s=0.2)
        request = Request(0, 10.0, 10.15, 0, 0, (3, 224, 224), "float32")
        self.assertEqual(
            collection_wait_s([request], [request], 10.0, settings, backend, 10.0), 0,
        )

    def test_selection_uses_earliest_deadline_then_priority(self):
        requests = [
            Request(0, 1, 5, 0, 0, (3, 224, 224), "float32"),
            Request(1, 1, 3, 0, 1, (3, 224, 224), "float32"),
            Request(2, 1, 3, 1, 2, (3, 224, 224), "float32"),
        ]
        self.assertEqual([row.id for row in select_batch(requests, "fifo", 4)], [0])
        self.assertEqual([row.id for row in select_batch(requests, "deadline", 2)], [2, 1])

    def test_percentiles_are_interpolated_and_empty_is_explicit(self):
        self.assertIsNone(percentile([], 0.95))
        self.assertEqual(percentile([10], 0.99), 10)
        self.assertEqual(percentile([0, 10, 20, 30], 0.5), 15)

    def test_replay_rejects_invalid_schedule_before_starting_worker(self):
        invalid = [
            [planned(0), planned(0)],
            [planned(0, 0.2), planned(1, 0.1)],
            [planned(0, -0.01)],
            [planned(0, math.nan)],
            [planned(True)],
            [planned(0, deadline_ms=math.nan)],
            [planned(0, deadline_ms=0)],
            [planned(0, deadline_ms=True)],
            [planned(0, input_index=-1)],
            [planned(0, shape=[3, 224, 224])],
            [planned(0, shape=(3, 0, 224))],
            [planned(0, shape=(3, True, 224))],
            [planned(0, dtype="")],
            [planned(0, priority="urgent")],
        ]
        for schedule in invalid:
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                replay(config(), schedule, "fifo", FakeBackend())

    def test_cancel_accounts_for_inflight_queued_and_future(self):
        settings = config(queue_capacity=16)
        gate = threading.Event()
        cancel = threading.Event()
        backend = FakeBackend(service_s=0.001, gate=gate)
        schedule = [planned(0), planned(1), planned(2), planned(3, 0.8)]

        def stop_after_worker_starts():
            if backend.started.wait(1):
                time.sleep(0.02)
                cancel.set()
            gate.set()

        stopper = threading.Thread(target=stop_after_worker_starts)
        stopper.start()
        try:
            result = replay(settings, schedule, "fifo", backend, cancel=cancel)
        finally:
            gate.set()
            stopper.join(2)
        self.assertFalse(stopper.is_alive())
        self.assertConserved(result, schedule, settings.queue_capacity)
        self.assertEqual(result["summary"]["cancelled_in_flight"], 1)
        self.assertEqual(result["summary"]["cancelled_queued"], 2)
        self.assertEqual(result["summary"]["unsubmitted_cancelled"], 1)

    def test_cancel_before_start_submits_nothing(self):
        settings = config()
        cancel = threading.Event()
        cancel.set()
        schedule = [planned(i, i * 0.02) for i in range(4)]
        backend = FakeBackend()
        result = replay(settings, schedule, "deadline", backend, cancel=cancel)
        self.assertConserved(result, schedule, settings.queue_capacity)
        self.assertEqual(result["summary"]["submitted"], 0)
        self.assertEqual(result["summary"]["unsubmitted_cancelled"], 4)
        self.assertFalse(backend.calls)

    def test_empty_schedule_reports_zero_with_no_latency_samples(self):
        result = replay(config(), [], "fifo", FakeBackend())
        self.assertConserved(result, [], 8)
        self.assertEqual(result["summary"]["completed"], 0)
        self.assertEqual(result["summary"]["deadline_success_fraction"], 0)
        self.assertEqual(result["summary"]["latency_sample_count"], 0)
        self.assertEqual(result["summary"]["latency_ms"], dict.fromkeys(("p50", "p95", "p99")))

    def test_backend_failure_is_terminal_and_does_not_hang(self):
        settings = config(queue_capacity=16)
        schedule = [planned(i, i * 0.001) for i in range(6)]
        result = replay(settings, schedule, "fifo", FakeBackend(fail=True))
        self.assertConserved(result, schedule, settings.queue_capacity)
        self.assertGreater(result["summary"]["failed"], 0)
        self.assertEqual(result["summary"]["failed_batch_count"], 1)
        self.assertEqual(result["summary"]["batch_count"], 1)
        batch = result["batches"][0]
        self.assertEqual(batch["status"], "failed")
        self.assertIsNone(batch["device_ms"])
        self.assertGreater(batch["service_ms"], 0)
        self.assertFalse(result["summary"]["device_timing_complete"])
        self.assertGreater(result["summary"]["service_ms_total"], 0)

    def test_broken_progress_callback_does_not_strand_or_cancel_requests(self):
        updates = []

        def broken(live):
            updates.append(live)
            raise RuntimeError("deliberate display failure")

        settings = config(queue_capacity=16)
        schedule = [planned(i, i * 0.001) for i in range(6)]
        result = replay(settings, schedule, "fifo", FakeBackend(), on_update=broken)
        self.assertConserved(result, schedule, settings.queue_capacity)
        self.assertEqual(result["summary"]["completed"], len(schedule))
        self.assertFalse(result["errors"])
        self.assertEqual(len(result["observer_errors"]), 1)
        self.assertEqual(len(updates), 1)

    def test_bad_prediction_count_is_a_failed_batch(self):
        class BrokenBackend(FakeBackend):
            def infer(self, requests):
                result = super().infer(requests)
                result["predictions"] = []
                return result

        settings = config()
        schedule = [planned(0)]
        result = replay(settings, schedule, "fifo", BrokenBackend())
        self.assertConserved(result, schedule, settings.queue_capacity)
        self.assertEqual(result["summary"]["failed"], 1)
        self.assertTrue(result["errors"])

    def test_missing_or_invalid_backend_timing_is_terminal_failure(self):
        missing = object()
        for key in ("device_ms", "service_ms"):
            for value in (missing, math.nan, math.inf, -1, True, "invalid"):
                with self.subTest(key=key, value="missing" if value is missing else value):
                    class BrokenTimingBackend(FakeBackend):
                        def infer(self, requests):
                            result = super().infer(requests)
                            if value is missing:
                                result.pop(key)
                            else:
                                result[key] = value
                            return result

                    settings = config()
                    schedule = [planned(0), planned(1)]
                    result = replay(settings, schedule, "fifo", BrokenTimingBackend())
                    self.assertConserved(result, schedule, settings.queue_capacity)
                    self.assertGreater(result["summary"]["failed"], 0)
                    self.assertEqual(result["summary"]["completed"], 0)
                    self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
