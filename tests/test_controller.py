"""Controller and export tests; no CUDA or model downloads."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from deadline_engine.experiment import export_report, summary_csv
from deadline_engine.controller import Controller
from test_core import FakeBackend, config, planned
from deadline_engine.core import replay


class ControllerTests(unittest.TestCase):
    def test_invalid_configuration_stays_idle(self):
        controller = Controller()
        for data in ({"arrival_rate": -5}, {"extra_field": 123}):
            with self.subTest(data=data), self.assertRaises((ValueError, TypeError)):
                controller.start(data)
        self.assertIsNone(controller.thread)
        self.assertEqual(json.loads(controller.encoded_state())["status"], "idle")

    def test_running_rejects_second_run_then_cancels_cleanly(self):
        started = threading.Event()

        def fake_run(settings, cancel, on_update, on_result):
            on_update("fifo", {"submitted": 1}, {"backend": "test-reference", "device": "cpu"})
            started.set()
            if not cancel.wait(2):
                raise RuntimeError("Test cancellation timeout")
            result = {"errors": [], "policy": "fifo"}
            on_result(result)
            return {"results": [result]}

        controller = Controller()
        with patch("deadline_engine.controller.run_comparison", fake_run):
            controller.start({"device": "cpu", "duration_s": 0.05})
            try:
                self.assertTrue(started.wait(1))
                self.assertEqual(json.loads(controller.encoded_state())["status"], "running")
                with self.assertRaisesRegex(ValueError, "already active"):
                    controller.start({"device": "cpu", "duration_s": 0.05})
                controller.stop()
            finally:
                controller.stop()
                controller.thread.join(2)
        self.assertFalse(controller.thread.is_alive())
        state = json.loads(controller.encoded_state())
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(state["current_policy"], "fifo")
        self.assertEqual(len(state["results"]), 1)

    def test_backend_exception_becomes_visible_failure(self):
        controller = Controller()
        with patch("deadline_engine.controller.run_comparison", side_effect=RuntimeError("weights are missing")):
            controller.start({"device": "cpu", "duration_s": 0.05})
            controller.thread.join(2)
        self.assertFalse(controller.thread.is_alive())
        state = json.loads(controller.encoded_state())
        self.assertEqual(state["status"], "failed")
        self.assertIn("weights are missing", state["error"])


class ExportTests(unittest.TestCase):
    def test_json_and_csv_preserve_real_reference_outcomes(self):
        settings = config()
        result = replay(settings, [planned(0), planned(1)], "fifo", FakeBackend())
        report = {"schema_version": 1, "results": [result]}
        scratch = Path(__file__).resolve().parents[1] / ".cache"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test-export-", dir=scratch) as folder:
            export_report(report, folder)
            saved = json.loads((Path(folder) / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["results"][0]["schedule_hash"], result["schedule_hash"])
            # DictReader silently skips blank rows, so inspect the bytes too.
            # In particular, Windows must not translate CSV's CRLF to CRCRLF.
            for name, expected_lines in (("comparison.csv", 2), ("requests.csv", 3)):
                raw = (Path(folder) / name).read_bytes()
                self.assertNotIn(b"\r\r\n", raw)
                self.assertEqual(len(raw.splitlines()), expected_lines)
                self.assertTrue(all(raw.splitlines()))
            with (Path(folder) / "requests.csv").open(newline="", encoding="utf-8") as handle:
                requests = list(csv.DictReader(handle))
            self.assertEqual([int(row["id"]) for row in requests], [0, 1])
            self.assertEqual([row["status"] for row in requests], [row["status"] for row in result["trace"]])
        summary = list(csv.DictReader(io.StringIO(summary_csv(report))))
        self.assertEqual(len(summary), 1)
        self.assertEqual(int(summary[0]["submitted"]), 2)
        self.assertEqual(summary[0]["schedule_hash"], result["schedule_hash"])
        self.assertEqual(summary[0]["policy"], "fifo")


if __name__ == "__main__":
    unittest.main()
