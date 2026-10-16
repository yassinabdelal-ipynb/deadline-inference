"""Native UI workflows and export boundaries; no model download or GPU needed."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

try:
    import tkinter as tk
except ImportError as exc:
    raise unittest.SkipTest("Optional native Tk support is unavailable: " + str(exc))

from deadline_engine.core import Config, TERMINAL, replay
from deadline_engine.desktop import APP_VERSION, DesktopApp, conservation_text, report_from_state, save_state
from test_core import FakeBackend, config, planned


def measured_result():
    return replay(config(), [planned(0), planned(1)], "fifo", FakeBackend())


def profile_report():
    stats = {"mean": 2.0, "stdev": .1, "p50": 2.0, "p95": 2.2, "min": 1.8, "max": 2.2}
    return {"kind": "device_profile", "schema_version": 2, "config": asdict(Config(device="cpu", max_batch=2)),
            "backend": {"device": "cpu", "model": "test", "parameter_count": 100},
            "cancelled": False, "errors": [], "notes": [],
            "rows": [{"batch_size": size, "sample_count": 10, "device_ms": stats, "service_ms": stats,
                      "images_per_second": stats, "sustained_images_per_second": size * 500,
                      "peak_memory": {}, "errors": []} for size in (1, 2)]}


def scratch_directory():
    scratch = Path(__file__).resolve().parents[1] / ".cache"
    scratch.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="test-desktop-", dir=scratch)


class DesktopExportTests(unittest.TestCase):
    def test_export_preserves_status_config_and_trace(self):
        result = measured_result()
        state = {"status": "cancelled", "config": result["config"], "results": [result]}
        with scratch_directory() as folder:
            path = Path(folder) / "measured results.json"
            save_state(state, path, "json")
            output = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(output["status"], "cancelled")
            self.assertEqual(output["config"], result["config"])
            self.assertEqual(output["results"][0]["trace"], json.loads(json.dumps(result["trace"])))
            save_state(state, path.with_suffix(".csv"), "csv")
            raw = path.with_suffix(".csv").read_bytes()
            self.assertNotIn(b"\r\r\n", raw)
            self.assertEqual(len(raw.splitlines()), 2)

    def test_no_results_cannot_export_and_accounting_detects_mismatch(self):
        with self.assertRaisesRegex(ValueError, "No measured results"):
            report_from_state({"results": []})
        result = measured_result()
        self.assertIn("2/2 OK", conservation_text([result]))
        result["summary"]["submitted"] += 1
        self.assertIn("MISMATCH", conservation_text([result]))

    def test_profile_exports_actual_device_measurements_without_fake_queue(self):
        state = {"status": "completed", "results": [], "profile": profile_report()}
        with scratch_directory() as folder:
            path = Path(folder) / "profile.json"
            save_state(state, path, "json")
            output = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(output["rows"], state["profile"]["rows"])
            self.assertEqual(output["kind"], "device_profile")
            self.assertNotIn("results", output)
            save_state(state, path.with_suffix(".csv"), "csv")
            raw = path.with_suffix(".csv").read_bytes()
            self.assertNotIn(b"\r\r\n", raw)
            self.assertEqual(len(raw.splitlines()), 3)


class FakeController:
    def __init__(self):
        self.state = {"status": "idle", "backend": {}, "live": {}, "results": []}
        self.thread = None
        self.started = None
        self.stopped = False

    def start(self, data):
        self.started = data
        self.state.update(status="running", config=data, current_policy="fifo")

    def stop(self):
        self.stopped = True
        self.state["status"] = "cancelled"

    def start_profile(self, data):
        self.started = data
        self.state.update(status="running", config=data, current_policy="profile", results=[], profile=None)

    def snapshot(self):
        return dict(self.state)


class NativeDesktopTests(unittest.TestCase):
    def setUp(self):
        self.controller = FakeController()
        try:
            self.app = DesktopApp(self.controller, start_probe=False)
            self.app.update()
        except tk.TclError as exc:
            self.skipTest("A display with Tcl/Tk is required: " + str(exc))

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.destroy()

    def test_run_validation_controls_and_cancellation(self):
        self.assertEqual(set(self.app.variables), set(asdict(Config())))
        self.assertEqual(str(self.app.cancel_button["state"]), "disabled")
        self.app.variables["queue_capacity"].set("1.5")
        self.app.run()
        self.assertIsNone(self.controller.started)
        self.assertIn("integer", self.app.message.get())
        self.app.variables["queue_capacity"].set("4")
        self.app.run_button.invoke()
        self.assertEqual(self.controller.started["queue_capacity"], 4)
        self.assertEqual(str(self.app.run_button["state"]), "disabled")
        self.assertEqual(str(self.app.cancel_button["state"]), "normal")
        self.app.cancel_button.invoke()
        self.assertTrue(self.controller.stopped)
        self.assertEqual(str(self.app.run_button["state"]), "normal")
        self.assertIn("cancelled", self.app.header_status.get())

    def test_measured_table_chart_export_and_resizing(self):
        result = measured_result()
        self.controller.state.update(status="completed", results=[result], config=result["config"])
        self.app._render(self.controller.snapshot())
        sections = self.app.chart.master.master
        def allocate_chart():
            self.app.update()
            # Exercise the same draggable panes a user can resize. Native font
            # metrics and window-manager limits differ across desktop systems;
            # a fixed outer geometry cannot promise a particular chart height.
            height = sections.winfo_height()
            sections.sashpos(0, int(height * 0.25))
            sections.sashpos(1, int(height * 0.85))
            self.app.update()
            self.app._draw_chart()
        def geometry_details():
            return {"window": self.app.winfo_geometry(), "chart": self.app.chart.winfo_geometry(),
                    "screen": (self.app.winfo_screenwidth(), self.app.winfo_screenheight()),
                    "sections": sections.winfo_geometry(),
                    "sashes": [sections.sashpos(index) for index in (0, 1)]}
        # Aqua delivers Configure events through the full event loop. Allocate
        # chart space from actual container dimensions before checking drawing.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            allocate_chart()
            if self.app.chart.winfo_width() > 100 and self.app.chart.winfo_height() >= 80 and self.app.chart.find_all():
                break
            time.sleep(0.01)
        self.assertIn("OK", self.app.accounting.get())
        self.assertGreater(len(self.app.results_table.get_children()), 15)
        self.assertGreater(len(self.app.chart.find_all()), 0, geometry_details())
        self.assertEqual(str(self.app.json_button["state"]), "normal")
        with scratch_directory() as folder:
            path = Path(folder) / "desktop report.json"
            with patch("deadline_engine.desktop.filedialog.asksaveasfilename", return_value=str(path)):
                self.app.json_button.invoke()
            output = json.loads(path.read_text(encoding="utf-8"))
            summary = output["results"][0]["summary"]
            self.assertEqual(summary["submitted"], sum(summary[k] for k in TERMINAL))
        self.app.geometry("1000x690")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            allocate_chart()
            if self.app.chart.winfo_width() > 100 and self.app.chart.winfo_height() > 60:
                break
            time.sleep(0.01)
        self.assertGreater(self.app.chart.winfo_width(), 100, geometry_details())
        self.assertGreater(self.app.chart.winfo_height(), 60, geometry_details())

    def test_close_requests_cancellation_without_blocking(self):
        self.app.run()
        self.app.close()
        self.assertTrue(self.app.closing)
        self.assertTrue(self.controller.stopped)
        self.assertEqual(str(self.app.run_button["state"]), "disabled")

    def _finish_background(self):
        thread = self.app.task_thread
        self.assertIsNotNone(thread)
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.app.after_cancel(self.app._poll_timer)
        self.app._poll()

    def _detected_runtime(self, **overrides):
        runtime = {"device": "cpu", "device_name": "CPU", "torch_version": "2.12.0+cpu",
                   "torchvision_version": "0.27.0+cpu", "weights_cached": True,
                   "cuda_available": False, "cuda_build": False, "torch_cuda_version": None,
                   "runtime_kind": "bundled", "executable": "DeadlineInference.exe"}
        runtime.update(overrides)
        self.app.tasks.put(("detect", runtime, None))
        self.app.after_cancel(self.app._poll_timer)
        self.app._poll()
        return runtime

    def test_cpu_bundle_offers_only_cpu_and_explains_gpu_launcher(self):
        self._detected_runtime()
        self.assertEqual(tuple(self.app.device_widget["values"]), ("cpu",))
        self.assertEqual(self.app.variables["device"].get(), "cpu")
        self.assertIn("CPU-only PyTorch", self.app.device_hint.get())
        self.assertIn("Run Deadline Inference.cmd", self.app.device_hint.get())
        self.assertIn(APP_VERSION + " / CPU runtime", self.app.title())
        self.app.run_button.invoke()
        self.assertEqual(self.controller.started["device"], "cpu")

    def test_cuda_runtime_offers_both_devices_and_defaults_to_gpu(self):
        self._detected_runtime(device="cuda", device_name="Test NVIDIA GPU", cuda_available=True,
                               cuda_build=True, torch_cuda_version="13.0", runtime_kind="source",
                               torch_version="2.12.0+cu130")
        self.assertEqual(tuple(self.app.device_widget["values"]), ("cpu", "cuda"))
        self.assertEqual(self.app.variables["device"].get(), "cuda")
        self.assertIn("Test NVIDIA GPU", self.app.device_hint.get())
        self.assertIn(APP_VERSION + " / GPU / CUDA", self.app.title())
        self.app.profile_button.invoke()
        self.assertEqual(self.controller.started["device"], "cuda")
        self.assertEqual(self.controller.state["current_policy"], "profile")

    def test_unavailable_cuda_cannot_start_replay_or_profile_even_if_set_manually(self):
        for cuda_build, expected in ((False, "CPU-only PyTorch"),
                                     (True, "no usable NVIDIA CUDA device")):
            with self.subTest(cuda_build=cuda_build):
                self._detected_runtime(cuda_build=cuda_build,
                                       torch_cuda_version="13.0" if cuda_build else None)
                self.app.variables["device"].set("cuda")
                for action in (self.app.run, self.app.profile):
                    action()
                    self.assertIsNone(self.controller.started)
                    self.assertEqual(self.controller.state["status"], "idle")
                    self.assertIn(expected, self.app.message.get())

    def test_import_validates_before_committing_and_keeps_prior_model_on_failure(self):
        prior = {"path": "previous checkpoint.pth", "sha256": "a" * 64}
        self.app.variables["weights_path"].set(prior["path"])
        self.app.imported_identity = prior
        self.app.weights_source.set("Imported weights")
        self.app.variables["num_classes"].set("3")
        with patch("deadline_engine.desktop.filedialog.askopenfilename", return_value="different weights.pth"), \
             patch("deadline_engine.backend.import_weights", side_effect=ValueError("Checkpoint keys do not match ResNet18")) as importer:
            self.app.import_model()
            self.assertEqual(str(self.app.run_button["state"]), "disabled")
            self._finish_background()
        importer.assert_called_once_with("different weights.pth", "mobilenet_v3_small", num_classes=3, image_size=224)
        self.assertEqual(self.app.variables["weights_path"].get(), prior["path"])
        self.assertIs(self.app.imported_identity, prior)
        self.assertEqual(self.app.weights_source.get(), "Imported weights")
        self.assertIn("Checkpoint keys", self.app.message.get())
        self.assertEqual(str(self.app.run_button["state"]), "normal")

    def test_successful_import_and_architecture_switch_reset_weight_identity(self):
        imported = {"path": "managed model.pth", "sha256": "b" * 64}
        self.app.variables["num_classes"].set("3")
        with patch("deadline_engine.desktop.filedialog.askopenfilename", return_value="my weights.pth"), \
             patch("deadline_engine.backend.import_weights", return_value=imported):
            self.app.import_button.invoke()
            self._finish_background()
        self.assertEqual(self.app.variables["weights_path"].get(), imported["path"])
        self.assertEqual(self.app.read_config()["num_classes"], 3)
        self.assertIn("origin/accuracy unverified", self.app.weights_state.get())
        self.app.model_label.set("ResNet18")
        self.app._model_changed()
        self.assertEqual(self.app.variables["model_id"].get(), "resnet18")
        self.assertEqual(self.app.variables["weights_path"].get(), "")
        self.assertEqual(self.app.read_config()["num_classes"], 1000)
        self.assertIsNone(self.app.imported_identity)

    def test_fetch_uses_selected_architecture_and_success_selects_official_weights(self):
        self.app.model_label.set("SqueezeNet1.1")
        self.app._model_changed()
        with patch("deadline_engine.backend.fetch_weights", return_value={"path": "official.pth"}) as fetcher:
            self.app.fetch_button.invoke()
            self._finish_background()
        fetcher.assert_called_once_with(model_id="squeezenet1_1")
        self.assertEqual(self.app.weights_source.get(), "Official pretrained")
        self.assertEqual(self.app.variables["weights_path"].get(), "")

    def test_device_profile_controls_cancel_and_render_separate_timings(self):
        self.app.profile_button.invoke()
        self.assertEqual(self.controller.state["current_policy"], "profile")
        self.assertEqual(str(self.app.profile_button["state"]), "disabled")
        self.assertEqual(self.app.live_variables["submitted"].get(), "--")
        self.app.cancel_button.invoke()
        self.assertTrue(self.controller.stopped)
        self.controller.state.update(status="completed", profile=profile_report())
        self.app._render(self.controller.snapshot())
        self.app.update_idletasks()
        self.assertEqual(self.app.results_table.heading("deadline")["text"], "Batch 2")
        values = [self.app.results_table.item(row)["values"] for row in self.app.results_table.get_children()]
        self.assertTrue(any(row[0] == "CPU forward p95 / ms" for row in values))
        self.assertTrue(any(row[0] == "Images per second / sustained" and float(row[2].replace(",", "")) == 1000 for row in values))
        self.assertIn("No queue", self.app.accounting.get())
        self.assertEqual(str(self.app.json_button["state"]), "normal")

    def test_recorded_model_identity_survives_changes_to_selected_model(self):
        result = measured_result()
        result["backend"].update(model="mobilenet_v3_small", model_label="MobileNetV3-Small",
                                 weights_kind="official_pretrained", weights_sha256="a" * 64,
                                 input_shape=[3, 224, 224])
        self.controller.state.update(status="completed", results=[result], config=result["config"])
        self.app._render(self.controller.snapshot())
        self.app.model_label.set("ResNet18")
        self.app._model_changed()
        self.app._render(self.controller.snapshot())
        values = {row[0]: row[1:] for row in (self.app.results_table.item(item)["values"] for item in self.app.results_table.get_children())}
        self.assertEqual(values["Measured image model"][0], "MobileNetV3-Small")
        self.assertEqual(values["Checkpoint SHA256 / prefix"][0], "a" * 16 + "...")
        self.assertEqual(values["Weights source"][0], "Official pretrained")
        self.assertIn("Inputs changed", self.app.accounting.get())
        exported = report_from_state(self.controller.snapshot())
        self.assertEqual(exported["results"][0]["backend"]["weights_sha256"], "a" * 64)
        self.assertEqual(exported["config"]["model_id"], "mobilenet_v3_small")

    def test_profile_loading_and_early_failure_keep_device_test_headings(self):
        settings = asdict(Config(device="cpu", max_batch=4))
        self.controller.state.update(status="loading", current_policy="profile", config=settings,
                                     backend={"device": "cpu"}, profile=None, results=[])
        self.app._render(self.controller.snapshot())
        self.assertEqual(self.app.results_table.heading("fifo")["text"], "Batch 1")
        self.assertEqual(self.app.results_table.heading("deadline")["text"], "Batch 4")
        self.assertIn("Warmup is excluded", self.app.accounting.get())
        self.assertNotIn("replays", self.app.accounting.get())
        self.assertEqual(str(self.app.json_button["state"]), "disabled")
        self.controller.state.update(status="failed", error="Missing weights")
        self.app._render(self.controller.snapshot())
        self.assertEqual(self.app.accounting.get(), "No device-test measurements are available.")
        self.assertEqual(self.app.results_table.heading("deadline")["text"], "Batch 4")


if __name__ == "__main__":
    unittest.main()
