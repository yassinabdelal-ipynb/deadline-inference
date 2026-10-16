"""Device-test accounting and cancellation without model downloads."""
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from deadline_engine.core import Config, PlannedRequest, make_schedule, replay
from deadline_engine.controller import Controller
from deadline_engine.profiling import run_profile, export_profile, profile_csv
from deadline_engine.errors import PreparationCancelled
from test_core import FakeBackend


class ProfileTests(unittest.TestCase):
    def settings(self, **values):
        return Config(device="cpu", max_batch=2, duration_s=.1, **values)

    def test_exact_measured_counts_exclude_warmup(self):
        backend = FakeBackend(service_s=.001)
        report = run_profile(self.settings(), samples=3, backend=backend)
        self.assertFalse(report["errors"])
        self.assertEqual([r["batch_size"] for r in report["rows"]], [1, 2])
        self.assertEqual([r["sample_count"] for r in report["rows"]], [3, 3])
        self.assertEqual(len(backend.calls), 10)  # 2 warmups +3 samples for each size
        self.assertEqual(report["rows"][1]["sustained_images_per_second"], 2000)
        self.assertEqual(report["rows"][1]["device_ms"]["stdev"], 0)

    def test_single_batch_one_is_not_duplicated(self):
        report = run_profile(replace(self.settings(), max_batch=1), samples=3, backend=FakeBackend(.001))
        self.assertEqual(len(report["rows"]), 1)

    def test_cancel_before_samples_still_exports_identity_and_cancellation(self):
        cancel = threading.Event()
        cancel.set()
        backend = FakeBackend(.001)
        report = run_profile(self.settings(), cancel=cancel, samples=3, backend=backend)
        rows = list(csv.DictReader(io.StringIO(profile_csv(report))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "mobilenet_v3_small")
        self.assertEqual(rows[0]["measurement_status"], "cancelled")
        self.assertEqual(rows[0]["sample_count"], "0")
        self.assertEqual(rows[0]["cancelled"], "True")
        self.assertEqual(backend.calls, [])

    def test_cancel_preparation_is_not_an_inference_failure(self):
        cancel = threading.Event()
        class PreparingBackend(FakeBackend):
            def prepare(self):
                cancel.set()
                raise PreparationCancelled("Stopped during warmup")
        report = run_profile(self.settings(), cancel=cancel, samples=3, backend=PreparingBackend())
        self.assertTrue(report["cancelled"])
        self.assertFalse(report["errors"])
        self.assertEqual(report["rows"], [])

    def test_scheduler_cancel_during_preparation_accounts_for_future_arrivals(self):
        cancel = threading.Event()
        class PreparingBackend(FakeBackend):
            def set_cancel_event(self, event):
                self.cancel = event

            def prepare(self):
                self.cancel.set()
                raise PreparationCancelled("Stopped during warmup")
        plans = [PlannedRequest(0, .01, 1000, 0, 0)]
        result = replay(self.settings(), plans, "fifo", PreparingBackend(), cancel=cancel)
        self.assertFalse(result["errors"])
        self.assertEqual(result["summary"]["submitted"], 0)
        self.assertEqual(result["summary"]["unsubmitted_cancelled"], 1)
        self.assertEqual(result["trace"][0]["status"], "cancelled_before_arrival")

    def test_early_timed_wait_does_not_admit_before_planned_arrival(self):
        class EarlyEvent(threading.Event):
            def __init__(self):
                super().__init__()
                self.early = True

            def wait(self, timeout=None):
                if self.early and timeout and timeout > 0:
                    self.early = False
                    return False  # Windows may wake before a requested timeout.
                return super().wait(timeout)
        result = replay(self.settings(), [PlannedRequest(0, .02, 1000, 0, 0)],
                        "fifo", FakeBackend(.001), cancel=EarlyEvent())
        row = result["trace"][0]
        self.assertGreaterEqual(row["submission_s"], row["arrival_s"])
        self.assertGreaterEqual(row["generator_lateness_ms"], 0)

    def test_cancel_during_samples_preserves_observed_work(self):
        cancel = threading.Event()
        def observe(live, backend):
            if live["profile_sample"] == 1:
                cancel.set()
        report = run_profile(self.settings(), cancel=cancel, on_update=observe,
                             samples=5, backend=FakeBackend(.001))
        self.assertTrue(report["cancelled"])
        self.assertEqual(len(report["rows"]), 1)
        self.assertEqual(report["rows"][0]["sample_count"], 1)

    def test_failure_is_not_counted_as_successful_sample(self):
        report = run_profile(self.settings(), samples=3, backend=FakeBackend(.001, fail=True))
        self.assertTrue(report["errors"])
        self.assertEqual(report["rows"][0]["sample_count"], 0)
        self.assertIsNone(report["rows"][0]["sustained_images_per_second"])

    def test_broken_observer_does_not_change_inference(self):
        def broken(*args):
            raise RuntimeError("widget failed")
        report = run_profile(self.settings(), samples=3, on_update=broken, backend=FakeBackend(.001))
        self.assertEqual(len(report["observer_errors"]), 1)
        self.assertEqual(sum(r["sample_count"] for r in report["rows"]), 6)

    def test_export_records_model_identity_samples_and_variability(self):
        report = run_profile(self.settings(), samples=3, backend=FakeBackend(.001))
        scratch = Path(__file__).resolve().parents[1] / ".cache"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="profile export ", dir=scratch) as folder:
            export_profile(report, folder)
            restored = json.loads((Path(folder)/"profile.json").read_text())
            rows = list(csv.DictReader(io.StringIO((Path(folder)/"profile.csv").read_text())))
        self.assertEqual(restored["kind"], "device_profile")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["sample_count"], "3")
        self.assertEqual(rows[0]["forward_stdev_ms"], "0.0")

    def test_configuration_preserves_new_input_metadata(self):
        config = self.settings(image_size=128, model_id="resnet18")
        self.assertTrue(all(p.shape == (3, 128, 128) for p in make_schedule(config)))
        for invalid in ({"model_id": "made_up"}, {"image_size": 513}, {"num_classes": 2}, {"weights_path": 123}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(config, **invalid).validate()

    def test_profile_controller_serializes_and_cancels(self):
        entered = threading.Event()
        def fake_profile(config, cancel, on_update):
            on_update({"profile_sample": 1}, {"device": "cpu"})
            entered.set()
            cancel.wait(2)
            return {"kind": "device_profile", "backend": {}, "rows": [], "errors": [], "cancelled": True}
        controller = Controller()
        with patch("deadline_engine.controller.run_profile", fake_profile):
            controller.start_profile({"device": "cpu"})
            try:
                self.assertTrue(entered.wait(1))
                with self.assertRaisesRegex(ValueError, "already active"):
                    controller.start({"device": "cpu"})
                controller.stop()
            finally:
                controller.stop()
                controller.thread.join(3)
        state = json.loads(controller.encoded_state())
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(state["profile"]["kind"], "device_profile")


if __name__ == "__main__":
    unittest.main()
