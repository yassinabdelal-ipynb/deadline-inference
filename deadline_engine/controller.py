"""UI-independent scheduling and device-test lifecycle."""
from dataclasses import asdict
import json
import threading

from .core import Config, make_schedule
from .experiment import run_comparison
from .profiling import run_profile


class Controller:
    def __init__(self):
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.state = {"status": "idle", "error": None, "backend": {}, "current_policy": None,
                      "live": {}, "results": [], "profile": None}

    def start(self, data):
        self._start(data, profile=False)

    def start_profile(self, data):
        self._start(data, profile=True)

    def _start(self, data, *, profile):
        config = Config(**data).validate()
        if not profile:
            make_schedule(config)
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("A run is already active. Cancel it or wait for completion.")
            self.cancel = threading.Event()
            self.state = {"status": "loading", "error": None, "backend": {"device": config.device},
                          "current_policy": "profile" if profile else None, "live": {}, "results": [],
                          "profile": None, "config": asdict(config), "mode": "profile" if profile else "scheduling"}
            self.thread = threading.Thread(target=self._run_profile if profile else self._run,
                                           args=(config,), name="inference-profile" if profile else "experiment", daemon=False)
            self.thread.start()

    def _run_profile(self, config):
        def update(live, backend):
            with self.lock:
                self.state.update(live=live, backend=backend,
                                  status="cancelling" if self.cancel.is_set() else "running")
        try:
            report = run_profile(config, cancel=self.cancel, on_update=update)
            with self.lock:
                self.state.update(profile=report, backend=report.get("backend", {}),
                                  status="failed" if report["errors"] else "cancelled" if report["cancelled"] else "completed",
                                  error="; ".join(report["errors"]) if report["errors"] else None)
        except Exception as exc:
            with self.lock:
                self.state.update(status="failed", error=str(exc))

    def _run(self, config):
        def update(policy, live, backend):
            with self.lock:
                self.state.update(current_policy=policy, live=live, backend=backend,
                                  status="cancelling" if self.cancel.is_set() else "running")
        def result(value):
            with self.lock:
                self.state["results"].append(value)
        try:
            report = run_comparison(config, self.cancel, update, result)
            errors = [e for r in report["results"] for e in r["errors"]]
            with self.lock:
                self.state.update(status="failed" if errors else "cancelled" if self.cancel.is_set() else "completed",
                                  error="; ".join(errors) if errors else None)
        except Exception as exc:
            with self.lock:
                self.state.update(status="failed", error=str(exc))

    def stop(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                self.cancel.set()
                self.state["status"] = "cancelling"

    def snapshot(self):
        with self.lock:
            return {**self.state, "live": dict(self.state["live"]), "results": list(self.state["results"])}

    def encoded_state(self):
        return json.dumps(self.snapshot(), allow_nan=False).encode()
