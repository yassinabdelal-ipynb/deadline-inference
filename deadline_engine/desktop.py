"""Native, local desktop workspace. No web browser or listening server is used.

Tk widgets belong exclusively to the main thread. Model work, diagnostics and
device detection run in background threads; a queue delivers their results.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, font, messagebox, ttk

from .core import Config, TERMINAL
from .controller import Controller
from .experiment import summary_csv
from .models import list_models

WHITE = "#FFFFFF"
PAPER = "#F5F5F5"
INK = "#202020"
MUTED = "#626262"
LINE = "#D8D8D8"
ACTIVE = {"loading", "running", "cancelling"}
POLICIES = {"fifo": "FIFO / batch 1", "deadline": "Deadline / batching"}
from . import __version__ as APP_VERSION


def display_number(value, digits=0):
    if value is None or isinstance(value, bool):
        return "--"
    try:
        number = float(value)
        if not math.isfinite(number):
            return "--"
        return f"{number:,.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def format_percent(value):
    return "--" if value is None else display_number(value * 100, 1) + "%"


def conservation_text(results):
    if not results:
        return "No measurements yet. Run replays to compare the same arrivals."
    checks = []
    for result in results:
        summary = result.get("summary", {})
        terminal = sum(summary.get(key, 0) for key in TERMINAL)
        submitted = summary.get("submitted", 0)
        label = result.get("policy", "unknown")
        checks.append(f"{label}: {terminal}/{submitted}" + (" OK" if terminal == submitted else " MISMATCH"))
    return "Request accounting | " + "   ".join(checks)


def report_from_state(state):
    """Keep export self-describing, including partial or cancelled outcomes."""
    if state.get("profile"):
        return {**state["profile"], "status": state.get("status"),
                "created_utc": datetime.now(timezone.utc).isoformat()}
    if not state.get("results"):
        raise ValueError("No measured results are available to save yet.")
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": state.get("config"), "status": state.get("status"),
            "results": state["results"],
            "notes": ["Synthetic images measure scheduling, not prediction accuracy.",
                      "Latency quantiles include completed requests only; deadline success uses all submitted requests.",
                      "Policies run sequentially on the same device with identical planned arrivals."]}


def save_state(state, destination, kind):
    """Export without opening a dialog, also usable by automated verification."""
    report = report_from_state(state)
    destination = Path(destination)
    if kind == "json":
        destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    elif kind == "csv":
        if report.get("kind") == "device_profile":
            from .profiling import profile_csv
            content = profile_csv(report)
        else:
            content = summary_csv(report)
        destination.write_text(content, encoding="utf-8", newline="")
    else:
        raise ValueError("Export format must be json or csv.")
    return destination


def probe_runtime():
    """Detect installed capabilities without loading a model or downloading files."""
    from .backend import _weight_spec
    torch, torchvision, weights, weight_path, _ = _weight_spec()
    cuda = torch.cuda.is_available()
    return {"device": "cuda" if cuda else "cpu",
            "device_name": torch.cuda.get_device_name(0) if cuda else "CPU",
            "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
            "weights_cached": weight_path.is_file(), "weights_path": str(weight_path),
            "weights": str(weights), "cuda_available": cuda,
            "torch_cuda_version": torch.version.cuda,
            "cuda_build": torch.version.cuda is not None,
            "runtime_kind": "bundled" if getattr(sys, "frozen", False) else "source",
            "executable": sys.executable}


def unavailable_cuda_message(runtime):
    """Describe the actual runtime limitation without blaming working hardware."""
    if not runtime.get("cuda_build", runtime.get("torch_cuda_version") is not None):
        if runtime.get("runtime_kind") == "bundled":
            return ("This standalone app includes CPU-only PyTorch and cannot use CUDA. "
                    "For NVIDIA GPU testing, use Run Deadline Inference.cmd in the source "
                    "project with CUDA dependencies installed. CPU testing is available here.")
        return ("This project environment has CPU-only PyTorch. Install the CUDA dependencies "
                "with scripts/setup.ps1 -Backend cuda, then reopen Run Deadline Inference.cmd. "
                "CPU testing is available now.")
    return ("CUDA-enabled PyTorch is installed, but no usable NVIDIA CUDA device was detected. "
            "Open System check for hardware and driver details. CPU testing is available.")


class DesktopApp(tk.Tk):
    def __init__(self, controller=None, *, start_probe=True):
        super().__init__()
        self.title(f"Deadline inference {APP_VERSION}")
        self.geometry("1320x900")
        self.minsize(1000, 690)
        self.configure(background=WHITE)
        self.controller = controller or Controller()
        self.tasks = queue.Queue()
        self.task_thread = None
        self.task_name = None
        self.checking = start_probe
        self.closing = False
        self.last_status = None
        self.last_policy = None
        self.result_key = None
        self.chart_key = None
        self.chart_series = []
        self.runtime = {}
        self.model_records = list_models()
        self.model_labels = {record["id"]: record["label"] for record in self.model_records}
        self.imported_identity = None
        self.pending_model = None
        self.profile_mode = False
        self.render_signature = None
        self.latest_state = self.controller.snapshot()
        self._style()
        self._build()
        for variable in self.variables.values():
            variable.trace_add("write", self._inputs_changed)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Control-Return>", self._run_shortcut)
        self.bind("<Escape>", self._cancel_shortcut)
        self.bind("<Control-s>", self._save_shortcut)
        self.log("Ready. Select an image model, then run a scheduling comparison or device test.")
        self.log("Inputs are procedural images. Predictions do not measure model accuracy.")
        self._render(self.latest_state)
        self._poll_timer = self.after(100, self._poll)
        if start_probe:
            self.after(40, lambda: self._background("detect", probe_runtime))

    def _style(self):
        names = set(font.families(self))
        self.family = "Consolas" if "Consolas" in names else "Menlo" if "Menlo" in names else font.nametofont("TkFixedFont").actual("family")
        self.mono = (self.family, 10)
        self.small = (self.family, 9)
        self.option_add("*Font", self.mono)
        self.option_add("*Background", WHITE)
        self.option_add("*Foreground", INK)
        self.option_add("*selectBackground", "#DADADA")
        self.option_add("*selectForeground", INK)
        self.style = ttk.Style(self)
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")
        self.style.configure(".", background=WHITE, foreground=INK, font=self.mono)
        self.style.configure("TFrame", background=WHITE)
        self.style.configure("TLabel", background=WHITE, foreground=INK)
        self.style.configure("Dim.TLabel", foreground=MUTED, font=self.small)
        self.style.configure("Section.TLabel", font=(self.family, 10, "bold"))
        self.style.configure("TButton", padding=(10, 7), borderwidth=1, relief="flat", background=PAPER, foreground=INK)
        self.style.map("TButton", background=[("disabled", PAPER), ("pressed", LINE), ("active", "#E8E8E8")], foreground=[("disabled", "#8A8A8A")], bordercolor=[("focus", INK)])
        self.style.configure("Run.TButton", background=INK, foreground=WHITE)
        self.style.map("Run.TButton", background=[("disabled", LINE), ("pressed", "#484848"), ("active", "#383838")], foreground=[("disabled", MUTED), ("!disabled", WHITE)])
        self.style.configure("TEntry", fieldbackground=WHITE, foreground=INK, bordercolor=LINE, padding=5, insertcolor=INK)
        self.style.map("TEntry", fieldbackground=[("disabled", PAPER)], bordercolor=[("focus", INK)])
        self.style.configure("TCombobox", fieldbackground=WHITE, foreground=INK, background=PAPER, padding=5, arrowcolor=INK)
        self.style.map("TCombobox", fieldbackground=[("readonly", WHITE), ("disabled", PAPER)], foreground=[("readonly", INK), ("disabled", MUTED)])
        self.style.configure("Treeview", background=WHITE, fieldbackground=WHITE, foreground=INK, rowheight=25, borderwidth=0)
        self.style.configure("Treeview.Heading", background=PAPER, foreground=INK, font=(self.family, 9, "bold"), relief="flat", padding=6)
        self.style.map("Treeview", background=[("selected", "#E2E2E2")], foreground=[("selected", INK)])
        self.style.configure("TPanedwindow", background=LINE)
        self.style.configure("Horizontal.TProgressbar", troughcolor=PAPER, background=MUTED, bordercolor=LINE, lightcolor=MUTED, darkcolor=MUTED)

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        header = ttk.Frame(self, padding=(18, 13))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="deadline inference", style="Section.TLabel").pack(side="left")
        ttk.Label(header, text="  /  Image classification models only", style="Dim.TLabel").pack(side="left")
        self.header_status = tk.StringVar(value="[ checking runtime ]")
        toolbar = ttk.Frame(self, padding=(18, 0, 18, 12))
        toolbar.grid(row=1, column=0, sticky="ew")
        self.run_button = ttk.Button(toolbar, text="> Run", command=self.run, style="Run.TButton")
        self.cancel_button = ttk.Button(toolbar, text="Stop / Esc", command=self.cancel)
        self.json_button = ttk.Button(toolbar, text="Save JSON", command=lambda: self.export("json"))
        self.csv_button = ttk.Button(toolbar, text="Save CSV", command=lambda: self.export("csv"))
        self.profile_button = ttk.Button(toolbar, text="Device test", command=self.profile)
        for button in (self.run_button, self.cancel_button, self.profile_button, self.json_button, self.csv_button):
            button.pack(side="left", padx=(0, 7))
        self.doctor_button = ttk.Button(toolbar, text="System check", command=self.doctor)
        self.fetch_button = ttk.Button(toolbar, text="Fetch model", command=self.fetch)
        self.doctor_button.pack(side="right")
        self.fetch_button.pack(side="right", padx=(0, 7))
        workspace = ttk.Panedwindow(self, orient="horizontal")
        workspace.grid(row=2, column=0, sticky="nsew", padx=18)
        controls = ttk.Frame(workspace, padding=(0, 0, 16, 0))
        workspace.add(controls, weight=0)
        self._build_controls(controls)
        instrument = ttk.Frame(workspace, padding=(16, 0, 0, 0))
        workspace.add(instrument, weight=1)
        instrument.columnconfigure(0, weight=1)
        instrument.rowconfigure(2, weight=1)
        self._build_live(instrument)
        self.message = tk.StringVar(value="Select a workload and run the same arrivals through both policies.")
        ttk.Label(instrument, textvariable=self.message, style="Dim.TLabel", wraplength=900, padding=(0, 9)).grid(row=1, column=0, sticky="ew")
        sections = ttk.Panedwindow(instrument, orient="vertical")
        sections.grid(row=2, column=0, sticky="nsew")
        comparison = ttk.Frame(sections)
        sections.add(comparison, weight=3)
        self._build_comparison(comparison)
        chart = ttk.Frame(sections)
        sections.add(chart, weight=2)
        self.chart_title = ttk.Label(chart, text="Latency / completed requests", style="Section.TLabel", padding=(0, 8))
        self.chart_title.pack(anchor="w")
        self.chart = tk.Canvas(chart, background=WHITE, highlightthickness=0, height=185, width=500)
        self.chart.pack(fill="both", expand=True)
        self.chart.bind("<Configure>", lambda _: self._draw_chart())
        logs = ttk.Frame(sections)
        sections.add(logs, weight=1)
        ttk.Label(logs, text="Log", style="Section.TLabel", padding=(0, 8)).pack(anchor="w")
        logframe = ttk.Frame(logs)
        logframe.pack(fill="both", expand=True)
        self.log_view = tk.Text(logframe, height=5, wrap="word", background=PAPER, foreground=INK, font=self.small, relief="flat", borderwidth=0, padx=9, pady=6, state="disabled", highlightthickness=1, highlightbackground=LINE)
        logscroll = ttk.Scrollbar(logframe, orient="vertical", command=self.log_view.yview)
        self.log_view.configure(yscrollcommand=logscroll.set)
        self.log_view.pack(side="left", fill="both", expand=True)
        logscroll.pack(side="right", fill="y")
        footer = ttk.Frame(self, padding=(18, 9))
        footer.grid(row=3, column=0, sticky="ew")
        self.footer = tk.StringVar(value="detecting runtime ...")
        ttk.Label(footer, textvariable=self.footer, style="Dim.TLabel").pack(side="left")
        ttk.Label(footer, text="Ctrl+Enter run   Esc stop   Ctrl+S save", style="Dim.TLabel").pack(side="right")

    def _build_controls(self, parent):
        ttk.Label(parent, text="Inputs", style="Section.TLabel", padding=(0, 0, 0, 10)).pack(anchor="w")
        scrollframe = ttk.Frame(parent)
        scrollframe.pack(fill="both", expand=True)
        canvas = tk.Canvas(scrollframe, width=272, highlightthickness=0, background=WHITE)
        scrollbar = ttk.Scrollbar(scrollframe, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        form = ttk.Frame(canvas, padding=(0, 0, 9, 0))
        window = canvas.create_window((0, 0), window=form, anchor="nw")
        form.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        # Scrolling stays confined to the workload panel rather than stealing
        # wheel input from the result table or the event log.
        def wheel(event):
            canvas.yview_scroll(-1 if event.delta > 0 or getattr(event, "num", None) == 4 else 1, "units")
            return "break"
        canvas.bind("<MouseWheel>", wheel)
        canvas.bind("<Button-4>", wheel)
        canvas.bind("<Button-5>", wheel)
        self.variables = {}
        self.input_widgets = []
        defaults = asdict(Config())
        defaults["device"] = "cpu"
        self.variables["weights_path"] = tk.StringVar(value=defaults.get("weights_path", ""))
        self.variables["model_id"] = tk.StringVar(value=defaults.get("model_id", "mobilenet_v3_small"))
        ttk.Label(form, text="image model architecture", style="Dim.TLabel").pack(anchor="w", pady=(9, 3))
        self.model_label = tk.StringVar(value=self.model_labels[self.variables["model_id"].get()])
        self.model_widget = ttk.Combobox(form, textvariable=self.model_label, values=list(self.model_labels.values()), state="readonly", width=24)
        self.model_widget.pack(fill="x")
        self.model_widget.bind("<<ComboboxSelected>>", self._model_changed)
        self.input_widgets.append(self.model_widget)
        self.weights_source = tk.StringVar(value="Official pretrained")
        ttk.Label(form, text="weights source", style="Dim.TLabel").pack(anchor="w", pady=(9, 3))
        self.source_widget = ttk.Combobox(form, textvariable=self.weights_source,
                                          values=("Official pretrained", "Imported weights"), state="readonly", width=24)
        self.source_widget.pack(fill="x")
        self.source_widget.bind("<<ComboboxSelected>>", self._source_changed)
        self.input_widgets.append(self.source_widget)
        self.import_button = ttk.Button(form, text="Import weights...", command=self.import_model)
        self.import_button.pack(fill="x", pady=(6, 0))
        self.weights_state = tk.StringVar(value="official weights / checked before execution")
        ttk.Label(form, textvariable=self.weights_state, style="Dim.TLabel", wraplength=235).pack(anchor="w", pady=(5, 4))
        ttk.Label(form, text="Import .pth/.pt state dictionaries for the selected architecture. Predictions do not measure accuracy.",
                  style="Dim.TLabel", wraplength=235).pack(anchor="w", pady=(3, 4))
        fields = [
            ("num_classes", "output classes / imported model", None, "Official pretrained models use 1,000 classes."),
            ("image_size", "image width and height / pixels", None, "RGB / float32 procedural images."),
            ("policy", "policy", ("compare", "fifo", "deadline"), "Compare replays the same schedule twice."),
            ("device", "device", ("cpu", "cuda"), "Detected automatically; selection stays explicit."),
            ("pattern", "arrival pattern", ("steady", "bursty", "overload"), None),
            ("arrival_rate", "base arrival rate / requests per s", None, None),
            ("burst_intensity", "burst multiplier / 1 to 10", None, None),
            ("duration_s", "arrival duration / seconds", None, None),
            ("deadline_ms", "deadline / milliseconds", None, None),
            ("max_batch", "maximum batch / 1 to 16", None, None),
            ("collection_ms", "maximum collection wait / ms", None, None),
            ("queue_capacity", "queue capacity / requests", None, None),
            ("seed", "random seed", None, "Same settings + seed = same planned arrivals."),
        ]
        for key, label, choices, description in fields:
            ttk.Label(form, text=label, style="Dim.TLabel").pack(anchor="w", pady=(9, 3))
            variable = tk.StringVar(value=str(defaults[key]))
            self.variables[key] = variable
            if choices:
                widget = ttk.Combobox(form, textvariable=variable, values=choices, state="readonly", width=24, takefocus=True)
            else:
                widget = ttk.Entry(form, textvariable=variable, width=26)
            widget.pack(fill="x")
            self.input_widgets.append(widget)
            if key == "device":
                self.device_widget = widget
            if description:
                if key == "device":
                    self.device_hint = tk.StringVar(value=description)
                    ttk.Label(form, textvariable=self.device_hint, style="Dim.TLabel", wraplength=235).pack(anchor="w", pady=(3, 0))
                else:
                    ttk.Label(form, text=description, style="Dim.TLabel", wraplength=235).pack(anchor="w", pady=(3, 0))
        ttk.Separator(form).pack(fill="x", pady=(17, 10))
        ttk.Label(form, text="On-time success counts every submitted request. Dropping more requests alone does not make a policy better.", style="Dim.TLabel", wraplength=235).pack(anchor="w", pady=(5, 12))
        for child in form.winfo_children():
            child.bind("<MouseWheel>", wheel)
            child.bind("<Button-4>", wheel)
            child.bind("<Button-5>", wheel)

    def _build_live(self, parent):
        live = ttk.Frame(parent)
        live.grid(row=0, column=0, sticky="ew")
        ttk.Label(live, text="Live requests", style="Section.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))
        self.live_title = live.winfo_children()[0]
        self.live_variables = {}
        labels = [("queue_depth", "queued"), ("in_flight", "in flight"),
                  ("submitted", "submitted"), ("admitted", "admitted"),
                  ("completed_on_time", "on time"), ("completed_late", "late"),
                  ("rejected", "rejected"), ("expired", "expired")]
        for index, (key, label) in enumerate(labels):
            column, row = index % 4, (index // 4) * 2 + 1
            live.columnconfigure(column, weight=1)
            variable = tk.StringVar(value="0")
            self.live_variables[key] = variable
            ttk.Label(live, text=label, style="Dim.TLabel").grid(row=row, column=column, sticky="w", padx=(0, 16))
            ttk.Label(live, textvariable=variable, font=(self.family, 16, "bold")).grid(row=row + 1, column=column, sticky="w", pady=(0, 9))
        self.cancelled_live = tk.StringVar(value="cancelled queued 0 / in flight 0   |   failed 0")
        ttk.Label(live, textvariable=self.cancelled_live, style="Dim.TLabel").grid(row=5, column=0, columnspan=4, sticky="w")

    def _build_comparison(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        ttk.Label(parent, text="Results", style="Section.TLabel", padding=(0, 8)).grid(row=0, column=0, sticky="w")
        table = ttk.Frame(parent)
        table.grid(row=1, column=0, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        self.results_table = ttk.Treeview(table, columns=("metric", "fifo", "deadline"), show="headings", selectmode="browse", height=9)
        for name, text, width in (("metric", "Measurement", 310), ("fifo", "FIFO / batch 1", 170), ("deadline", "Deadline / batch", 175)):
            self.results_table.heading(name, text=text)
            self.results_table.column(name, width=width, minwidth=115 if name != "metric" else 255, anchor="w" if name == "metric" else "e", stretch=name == "metric")
        self.results_table.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(table, orient="vertical", command=self.results_table.yview)
        horizontal = ttk.Scrollbar(table, orient="horizontal", command=self.results_table.xview)
        self.results_table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.accounting = tk.StringVar(value=conservation_text([]))
        ttk.Label(parent, textvariable=self.accounting, style="Dim.TLabel", padding=(0, 5)).grid(row=2, column=0, sticky="w")

    def log(self, message):
        self.log_view.configure(state="normal")
        self.log_view.insert("end", time.strftime("%H:%M:%S") + "  > " + str(message) + "\n")
        # Logs are bounded too: long desktop sessions should not grow forever.
        if int(self.log_view.index("end-1c").split(".")[0]) > 400:
            self.log_view.delete("1.0", "101.0")
        self.log_view.see("end")
        self.log_view.configure(state="disabled")

    def _run_shortcut(self, _event):
        self.run()
        return "break"

    def _cancel_shortcut(self, _event):
        self.cancel()
        return "break"

    def _save_shortcut(self, _event):
        self.export("json")
        return "break"

    def read_config(self, overrides=None):
        data = {}
        integer_fields = {"max_batch", "queue_capacity", "seed", "num_classes", "image_size"}
        text_fields = {"policy", "device", "pattern", "model_id", "weights_path"}
        for key, variable in self.variables.items():
            value = variable.get().strip()
            try:
                data[key] = value if key in text_fields else int(value) if key in integer_fields else float(value)
            except ValueError as exc:
                raise ValueError(f"{key.replace('_', ' ')} must be {'an integer' if key in integer_fields else 'a number'}.") from exc
        if overrides:
            data.update(overrides)
        return asdict(Config(**data).validate())

    def _model_changed(self, _event=None):
        """Changing architecture explicitly returns to official weights."""
        if self._busy():
            return
        model_id = next((key for key, label in self.model_labels.items() if label == self.model_label.get()), None)
        if model_id is None:
            return
        self.variables["model_id"].set(model_id)
        self.variables["image_size"].set("224")
        self._use_official()
        self.message.set(f"Selected {self.model_labels[model_id]}. Fetch model downloads its official weights; Import weights uses your compatible checkpoint.")

    def _use_official(self):
        self.variables["weights_path"].set("")
        self.variables["num_classes"].set("1000")
        self.imported_identity = None
        self.weights_source.set("Official pretrained")
        self.weights_state.set("official weights / checked before execution")

    def _source_changed(self, _event=None):
        if self._busy():
            return
        if self.weights_source.get() == "Official pretrained":
            self._use_official()
        elif not self.variables["weights_path"].get():
            # Merely selecting a source must never manufacture a valid import.
            self.weights_source.set("Official pretrained")
            self.import_model()

    def import_model(self):
        if self._busy():
            return
        path = filedialog.askopenfilename(parent=self, title="Import image model weights for " + self.model_label.get(),
                                           filetypes=[("PyTorch weights", "*.pth *.pt"), ("All files", "*.*")])
        if not path:
            return
        try:
            config = self.read_config({"weights_path": path})
        except (ValueError, TypeError) as exc:
            self.message.set("INPUT ERROR | " + str(exc))
            self.log("Import not started: " + str(exc))
            return
        from .backend import import_weights
        model_id, num_classes, image_size = config["model_id"], config["num_classes"], config["image_size"]
        self.pending_model = model_id
        self._background("import", lambda: import_weights(path, model_id, num_classes=num_classes, image_size=image_size))

    def profile(self):
        if self._busy():
            return
        try:
            config = self.read_config()
            self._validate_device(config)
            self.controller.start_profile(config)
            self.last_status = None
            self.last_policy = None
            self.log(f"Device test requested: {self.model_labels[config['model_id']]} / {config['device']} / batch 1 and {config['max_batch']}.")
            self._render(self.controller.snapshot())
        except (ValueError, TypeError, RuntimeError) as exc:
            self.message.set("INPUT ERROR | " + str(exc))
            self.log("Device test not started: " + str(exc))

    def _busy(self):
        return self.checking or self.task_thread is not None or self.controller.snapshot().get("status") in ACTIVE or self.closing

    def _validate_device(self, config):
        # The readonly selector prevents normal unsupported choices. Check again
        # so shortcuts, stale values or programmatic edits cannot start CUDA work
        # in a CPU-only runtime. Backend checks remain the CLI's final authority.
        if config["device"] == "cuda" and self.runtime and not self.runtime.get("cuda_available"):
            raise ValueError(unavailable_cuda_message(self.runtime))

    def _inputs_changed(self, *_args):
        """Keep retained measurements visibly tied to their recorded settings."""
        if self._busy():
            return
        state = self.controller.snapshot()
        profile, results = state.get("profile"), state.get("results", [])
        measured = profile.get("config") if profile else results[0].get("config") if results else None
        if measured is None:
            return
        if profile:
            base = "Direct model calls after warmup. Host service includes input assembly and transfers. No queue or deadline comparison."
        else:
            base = conservation_text(results)
        try:
            changed = self.read_config() != measured
        except (ValueError, TypeError):
            changed = True
        self.accounting.set(base + (" | Inputs changed; displayed results belong to the recorded run." if changed else ""))

    def run(self):
        if self._busy():
            return
        try:
            config = self.read_config()
            self._validate_device(config)
            self.controller.start(config)
            self.log(f"Run requested: {config['policy']} / {config['device']} / {config['pattern']} / seed {config['seed']}.")
            self.last_status = None
            self.last_policy = None
            self._render(self.controller.snapshot())
        except (ValueError, TypeError, RuntimeError) as exc:
            self.message.set("INPUT ERROR | " + str(exc))
            self.log("Run not started: " + str(exc))

    def cancel(self):
        if self.controller.snapshot().get("status") not in ACTIVE:
            return
        self.controller.stop()
        self.log("Stop requested. Any active forward finishes safely; queued work is accounted for.")
        self._render(self.controller.snapshot())

    def export(self, kind):
        state = self.controller.snapshot()
        if self._busy() or not (state.get("results") or state.get("profile")):
            return
        name = "device-test" if state.get("profile") else "comparison"
        destination = filedialog.asksaveasfilename(parent=self, title=f"Save {name} {kind.upper()}", defaultextension="." + kind,
                                                  initialfile=name + "." + kind, filetypes=[(kind.upper(), "*." + kind)])
        if not destination:
            return
        try:
            save_state(state, destination, kind)
            self.message.set("Saved " + destination)
            self.log("Saved measured results: " + destination)
        except (OSError, ValueError, TypeError) as exc:
            self.message.set("SAVE FAILED | " + str(exc))
            self.log("Save failed: " + str(exc))

    def _background(self, name, work):
        if self.task_thread is not None:
            return
        self.task_name = name
        def task():
            try:
                self.tasks.put((name, work(), None))
            except Exception as exc:
                self.tasks.put((name, None, str(exc)))
        self.task_thread = threading.Thread(target=task, name="desktop-" + name, daemon=False)
        self.task_thread.start()
        self.log({"detect": "Detecting the installed runtime and device. No inference or downloads.",
                  "fetch": "Fetching/verifying official pretrained model weights. This may take a moment.",
                  "import": "Checking checkpoint compatibility on CPU and copying verified weights into the local model cache.",
                  "doctor": "System check reads this computer's Python, CPU, CUDA and GPU information."}[name])
        self._render(self.controller.snapshot())

    def fetch(self):
        if self._busy():
            return
        from .backend import fetch_weights
        model_id = self.variables["model_id"].get()
        self.pending_model = model_id
        self._background("fetch", lambda: fetch_weights(model_id=model_id))

    def doctor(self):
        if self._busy():
            return
        from .doctor import diagnose
        self._background("doctor", diagnose)

    def _show_doctor(self, data):
        window = tk.Toplevel(self)
        window.title("System check")
        window.geometry("850x620")
        window.minsize(500, 350)
        window.configure(background=WHITE)
        ttk.Label(window, text="Installed Python, PyTorch and available CPU/GPU tools. No settings are changed.", style="Dim.TLabel", padding=12, wraplength=780).pack(anchor="w")
        text = tk.Text(window, background=WHITE, foreground=INK, font=self.small, wrap="none", relief="flat", padx=12, pady=6)
        scroll = ttk.Scrollbar(window, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)
        text.insert("1.0", json.dumps(data, indent=2))
        text.configure(state="disabled")
        controls = ttk.Frame(window, padding=12)
        controls.pack(fill="x")
        def save():
            path = filedialog.asksaveasfilename(parent=window, title="Save system report", initialfile="system-check.json", defaultextension=".json", filetypes=[("JSON", "*.json")])
            if path:
                try:
                    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
                    self.log("Saved system check: " + path)
                except OSError as exc:
                    messagebox.showerror("Could not save diagnostics", str(exc), parent=window)
        ttk.Button(controls, text="Save JSON", command=save).pack(side="left")
        ttk.Button(controls, text="Close", command=window.destroy).pack(side="right")

    def _poll(self):
        try:
            while True:
                name, data, error = self.tasks.get_nowait()
                self.task_thread = None
                self.task_name = None
                self.checking = False
                if error:
                    self.log(f"{name} failed: {error}")
                    self.message.set("RUNTIME ERROR | " + error)
                    if name == "detect":
                        self.footer.set("Runtime unavailable / run setup and System check")
                        self.weights_state.set("weights: unable to check")
                elif name == "detect":
                    self.runtime = data
                    self.device_widget.configure(values=("cpu", "cuda") if data["cuda_available"] else ("cpu",))
                    self.variables["device"].set(data["device"])
                    runtime_label = "GPU / CUDA" if data["cuda_available"] else "CPU runtime"
                    self.title(f"Deadline inference {APP_VERSION} / {runtime_label}")
                    if data["cuda_available"]:
                        self.device_hint.set(f"CUDA available: {data['device_name']}. CPU is also available.")
                    else:
                        self.device_hint.set(unavailable_cuda_message(data))
                        self.log(unavailable_cuda_message(data))
                    self.weights_state.set("weights: cached / verified on Run" if data["weights_cached"] else "weights: missing / use Fetch model")
                    self.log(f"Detected {data['device'].upper()}: {data['device_name']} / PyTorch {data['torch_version']}.")
                    if not data["weights_cached"]:
                        self.message.set("Pretrained weights are missing. Fetch model explicitly downloads the official weights; then Run.")
                        self.log("Model weights are not cached. Fetch model is an explicit download; Run never downloads.")
                elif name == "fetch":
                    self._use_official()
                    self.weights_state.set("weights: cached / SHA256 checked")
                    self.runtime["weights_cached"] = True
                    self.log("Model weights ready: " + data["path"])
                    self.message.set("Pretrained weights verified. Ready to run.")
                elif name == "import":
                    # Commit the selected checkpoint only after validation and
                    # managed copy succeed. A failed import leaves prior state.
                    self.variables["weights_path"].set(data["path"])
                    self.imported_identity = data
                    self.weights_source.set("Imported weights")
                    digest = data.get("sha256", "unknown")
                    self.weights_state.set(f"imported: {Path(data['path']).name}\nSHA256: {digest[:16]}...\nCompatibility checked; origin/accuracy unverified.")
                    self.message.set("Imported image model weights validated. Ready to run or device test.")
                    self.log("Imported weights: " + data["path"] + " / SHA256 " + digest)
                elif name == "doctor":
                    self.log("System check completed. Run a replay or device test to verify model execution.")
                    if not self.closing:
                        self._show_doctor(data)
        except queue.Empty:
            pass
        state = self.controller.snapshot()
        self._render(state)
        if self.closing:
            thread = self.controller.thread
            if (thread is None or not thread.is_alive()) and self.task_thread is None:
                self.destroy()
                return
        self._poll_timer = self.after(100, self._poll)

    def _render(self, state):
        self.latest_state = state
        status = state.get("status", "idle")
        live_state = state.get("live", {})
        results_state = state.get("results", [])
        signature = (status, state.get("error"), state.get("current_policy"),
                     self.checking, self.task_name, self.task_thread is not None, self.closing,
                     tuple((r.get("policy"), id(r)) for r in results_state), id(state.get("profile")),
                     tuple(live_state.get(k) for k in (*self.live_variables, "cancelled_queued", "cancelled_in_flight", "failed", "profile_batch_size", "profile_sample", "profile_samples")),
                     len(live_state.get("trace", [])), state.get("backend", {}).get("device_name"),
                     self.runtime.get("device_name"))
        # Rewriting every widget each poll creates needless layout/redraw work
        # while idle. The poll still observes cancellation and worker progress.
        if signature == self.render_signature:
            return
        self.render_signature = signature
        busy = self.checking or self.task_thread is not None or status in ACTIVE or self.closing
        badge = "closing safely" if self.closing else "checking runtime" if self.checking else "fetching model" if self.task_name == "fetch" else "system check running" if self.task_name == "doctor" else status
        self.header_status.set("[ " + badge + " ]")
        self.run_button.configure(state="disabled" if busy else "normal", text="> Warming up" if status == "loading" else "> Running" if status == "running" else "> Stopping" if status == "cancelling" else "> Run")
        self.cancel_button.configure(state="normal" if status in {"loading", "running"} and not self.closing else "disabled")
        for button in (self.doctor_button, self.fetch_button, self.import_button, self.profile_button):
            button.configure(state="disabled" if busy else "normal")
        for button in (self.json_button, self.csv_button):
            button.configure(state="normal" if (state.get("results") or state.get("profile")) and not busy else "disabled")
        for widget in self.input_widgets:
            widget.configure(state="disabled" if busy else "readonly" if isinstance(widget, ttk.Combobox) else "normal")
        if status != self.last_status:
            if status != "idle":
                profiling = state.get("current_policy") == "profile" or state.get("mode") == "profile"
                messages = {"loading": "Loading selected weights; warmup/calibration are outside measurement.",
                            "running": "Device test running repeated model calls after warmup." if profiling else "Open-loop replay running. Arrivals are independent of previous responses.",
                            "cancelling": "Stopping safely. Every submitted request receives a terminal outcome.",
                            "completed": "Device test finished. Timings describe this model, batch size and computer; no general computing-power score is implied." if state.get("profile") else "Run finished. Compare success across all submissions and latency across completions.",
                            "cancelled": "Run cancelled. Partial results retain explicit cancellation categories.",
                            "failed": "Run failed. " + (state.get("error") or "Inspect the event log.")}
                text = messages.get(status, status)
                self.message.set(text)
                self.log(text)
            self.last_status = status
        policy = state.get("current_policy")
        if policy and policy != self.last_policy:
            self.log("Device test started." if policy == "profile" else "Measured replay started: " + POLICIES.get(policy, policy))
            self.last_policy = policy
        profiling = policy == "profile" or bool(state.get("profile"))
        self.live_title.configure(text="Device test" if profiling else "Live requests / " + (POLICIES.get(policy, policy) if policy else "waiting"))
        live = state.get("live", {})
        for key, variable in self.live_variables.items():
            variable.set("--" if profiling else display_number(live.get(key, 0)))
        self.cancelled_live.set(f"batch {live.get('profile_batch_size', '--')} / sample {live.get('profile_sample', 0)} of {live.get('profile_samples', 10)}" if profiling else f"cancelled queued {live.get('cancelled_queued', 0)} / in flight {live.get('cancelled_in_flight', 0)}   |   failed {live.get('failed', 0)}")
        backend = state.get("backend", {})
        info = backend if backend.get("device_name") else self.runtime
        if info.get("device_name"):
            self.footer.set(f"{str(info.get('device', 'cpu')).upper()} / {info['device_name']} / PyTorch {info.get('torch_version', '?')}")
        results = state.get("results", [])
        profile = state.get("profile")
        self.profile_mode = profiling
        self.chart_title.configure(text="Device timing / no request queue" if profiling else "Latency / completed requests")
        key = (tuple((r.get("policy"), id(r)) for r in results), id(profile) if profile else None,
               profiling, state.get("config", {}).get("max_batch"), backend.get("device"),
               status if profiling and profile is None else None)
        if key != self.result_key:
            self.result_key = key
            if profiling:
                self._render_profile(profile or {"config": state.get("config", {}), "backend": backend, "rows": []})
                if profile is None:
                    self.accounting.set("Device test in progress. Warmup is excluded; measurements appear when the test finishes." if status in ACTIVE else "No device-test measurements are available.")
            else:
                self._render_results(results)
            self._inputs_changed()
        live_trace = live.get("trace", [])
        chart_key = (key, policy, status, len(live_trace), live.get("completed_on_time"), live.get("completed_late"))
        if chart_key != self.chart_key:
            self.chart_key = chart_key
            self.chart_series = list(results)
            if status in {"running", "cancelling"} and policy and live_trace and not any(r.get("policy") == policy for r in results):
                offset = max(0, live.get("completed_on_time", 0) + live.get("completed_late", 0) - len(live_trace))
                self.chart_series.append({"policy": policy, "trace": live_trace, "live": True, "offset": offset})
            self._draw_chart()

    def _render_results(self, results):
        self.results_table.delete(*self.results_table.get_children())
        self.results_table.heading("fifo", text="FIFO / batch 1")
        self.results_table.heading("deadline", text="Deadline / batch")
        recorded = {result["policy"]: result for result in results}
        identities = [
            ("Measured image model", lambda r: self._recorded_model(r)),
            ("Weights source", lambda r: self._recorded_weights(r)),
            ("Checkpoint SHA256 / prefix", lambda r: self._recorded_hash(r)),
            ("Input / RGB float32", lambda r: self._recorded_input(r)),
        ]
        for label, formatter in identities:
            values = [formatter(recorded[policy]) if policy in recorded else "--" for policy in ("fifo", "deadline")]
            self.results_table.insert("", "end", values=(label, *values))
        rows = [
            ("Submitted", lambda s: display_number(s.get("submitted"))),
            ("Admitted", lambda s: display_number(s.get("admitted"))),
            ("Completed on time", lambda s: display_number(s.get("completed_on_time"))),
            ("Completed late", lambda s: display_number(s.get("completed_late"))),
            ("Rejected / queue full", lambda s: display_number(s.get("rejected"))),
            ("Expired / before inference", lambda s: display_number(s.get("expired"))),
            ("Cancelled / queued", lambda s: display_number(s.get("cancelled_queued"))),
            ("Cancelled / in flight", lambda s: display_number(s.get("cancelled_in_flight"))),
            ("Failed", lambda s: display_number(s.get("failed"))),
            ("Unsubmitted / cancelled plan", lambda s: display_number(s.get("unsubmitted_cancelled"))),
            ("Completion rate", lambda s: format_percent(s.get("completion_rate"))),
            ("On time / all submitted", lambda s: format_percent(s.get("deadline_success_fraction"))),
            ("Deadline misses / late + expired", lambda s: display_number(s.get("deadline_misses"))),
            ("Discarded / rejected + expired", lambda s: display_number(s.get("discarded"))),
            ("Throughput / completions per s", lambda s: display_number(s.get("throughput_rps"), 1)),
            ("Actual mean batch size", lambda s: display_number(s.get("mean_batch_size"), 2)),
            ("Forward time per image / ms", lambda s: display_number(s.get("forward_ms_per_image"), 3)),
            ("Queue wait p95 / ms", lambda s: display_number(s.get("queue_wait_ms", {}).get("p95"), 2)),
            *[(f"End-to-end latency {q} / ms", lambda s, q=q: display_number(s.get("latency_ms", {}).get(q), 2)) for q in ("p50", "p95", "p99")],
            ("Latency samples / completions", lambda s: display_number(s.get("latency_sample_count"))),
            ("Generator lateness p95 / ms", lambda s: display_number(s.get("generator_lateness_ms", {}).get("p95"), 2)),
            ("Peak CUDA allocated / MiB", lambda s: self._memory(s, "cuda_peak_allocated_bytes")),
            ("Peak CUDA reserved / MiB", lambda s: self._memory(s, "cuda_peak_reserved_bytes")),
            ("Process lifetime peak / MiB", lambda s: self._memory(s, "process_peak_resident_bytes")),
        ]
        by_policy = {r["policy"]: r.get("summary", {}) for r in results}
        for label, formatter in rows:
            values = [formatter(by_policy[policy]) if policy in by_policy else "--" for policy in ("fifo", "deadline")]
            self.results_table.insert("", "end", values=(label, *values))
        self.accounting.set(conservation_text(results))

    def _recorded_model(self, result):
        backend, config = result.get("backend", {}), result.get("config", {})
        model_id = backend.get("model", config.get("model_id"))
        return backend.get("model_label") or self.model_labels.get(model_id, model_id or "--")

    @staticmethod
    def _recorded_weights(result):
        kind = result.get("backend", {}).get("weights_kind")
        return {"official_pretrained": "Official pretrained", "imported_state_dict": "Imported weights"}.get(kind, kind or "--")

    @staticmethod
    def _recorded_hash(result):
        digest = result.get("backend", {}).get("weights_sha256")
        return str(digest)[:16] + "..." if digest else "--"

    @staticmethod
    def _recorded_input(result):
        shape = result.get("backend", {}).get("input_shape")
        if shape and len(shape) == 3:
            return f"{shape[1]} x {shape[2]}"
        size = result.get("config", {}).get("image_size")
        return f"{size} x {size}" if size else "--"

    def _render_profile(self, report):
        self.results_table.delete(*self.results_table.get_children())
        rows = report.get("rows", [])
        by_batch = {row.get("batch_size"): row for row in rows}
        last_batch = report.get("config", {}).get("max_batch", max(by_batch, default=1))
        batches = (1, last_batch)
        self.results_table.heading("fifo", text="Batch 1")
        self.results_table.heading("deadline", text="Batch " + str(last_batch))
        backend = report.get("backend", {})
        forward_label = "CUDA forward" if backend.get("device") == "cuda" else "CPU forward"
        metrics = [
            ("Samples / after warmup", lambda r: display_number(r.get("sample_count"))),
            *[(f"{forward_label} {stat} / ms", lambda r, stat=stat: display_number(r.get("device_ms", {}).get(stat), 3)) for stat in ("mean", "p95", "stdev")],
            *[(f"Host service {stat} / ms", lambda r, stat=stat: display_number(r.get("service_ms", {}).get(stat), 3)) for stat in ("mean", "p95", "stdev")],
            ("Images per second / sustained", lambda r: display_number(r.get("sustained_images_per_second"), 1)),
            ("Images per second / stdev", lambda r: display_number(r.get("images_per_second", {}).get("stdev"), 1)),
            ("Peak CUDA allocated / MiB", lambda r: self._memory(r, "cuda_peak_allocated_bytes")),
            ("Peak CUDA reserved / MiB", lambda r: self._memory(r, "cuda_peak_reserved_bytes")),
            ("Process lifetime peak / MiB", lambda r: self._memory(r, "process_peak_resident_bytes")),
        ]
        for label, formatter in metrics:
            values = [formatter(by_batch[batch]) if batch in by_batch else "--" for batch in batches]
            self.results_table.insert("", "end", values=(label, *values))
        config = report.get("config", {})
        for label, value in (("Architecture", self.model_labels.get(config.get("model_id"), config.get("model_id", "--"))),
                             ("Weights source", self._recorded_weights(report)),
                             ("Checkpoint SHA256 / prefix", self._recorded_hash(report)),
                             ("Input / RGB float32", f"{config.get('image_size', '--')} x {config.get('image_size', '--')}"),
                             ("Parameters", display_number(backend.get("parameter_count", backend.get("parameters"))))):
            self.results_table.insert("", "end", values=(label, value, value))
        self.accounting.set("Direct model calls after warmup. Host service includes input assembly and transfers. No queue or deadline comparison.")

    @staticmethod
    def _memory(summary, key):
        value = summary.get("peak_memory", {}).get(key)
        return "--" if value is None else display_number(value / 1048576, 1)

    def _draw_chart(self):
        self.chart.delete("all")
        width, height = self.chart.winfo_width(), self.chart.winfo_height()
        if width < 100 or height < 80:
            return
        series = []
        for result in self.chart_series:
            rows = [r for r in result.get("trace", []) if r.get("status") in {"completed_on_time", "completed_late"} and r.get("latency_ms") is not None]
            if rows and all(row.get("finish_s") is not None for row in rows):
                rows.sort(key=lambda row: row["finish_s"])
            if rows:
                series.append((result, rows))
        if not series:
            text = "Device test measures repeated model calls.\nRead CPU/CUDA timings and memory in the results table." if self.profile_mode else "No completed-request measurements yet.\nRun a replay to draw real latency traces."
            self.chart.create_text(width / 2, height / 2, text=text, fill=MUTED, font=self.small, justify="center")
            return
        left, top, right, bottom = 57, 27, width - 15, height - 36
        plot_width, plot_height = max(1, right - left), max(1, bottom - top)
        maximum = max(max(row["latency_ms"] for row in rows) for _, rows in series) * 1.12 or 1
        count = max(len(rows) + result.get("offset", 0) for result, rows in series)
        for step in range(4):
            y = top + plot_height * step / 3
            self.chart.create_line(left, y, right, y, fill=LINE)
            self.chart.create_text(left - 8, y, text=display_number(maximum * (1 - step / 3), 0), anchor="e", font=self.small, fill=MUTED)
        self.chart.create_text(6, 8, text="ms", anchor="nw", font=self.small, fill=MUTED)
        legend_x = left
        for result, rows in series:
            dashed = result["policy"] == "deadline"
            label = POLICIES.get(result["policy"], result["policy"]) + (" / live" if result.get("live") else "")
            self.chart.create_line(legend_x, 10, legend_x + 20, 10, fill=INK, dash=(5, 3) if dashed else (), width=1.5)
            self.chart.create_text(legend_x + 27, 10, text=label, anchor="w", font=self.small, fill=INK)
            legend_x += 250
            points = []
            for index, row in enumerate(rows):
                x = left + ((index + result.get("offset", 0)) / max(1, count - 1)) * plot_width
                y = top + (1 - row["latency_ms"] / maximum) * plot_height
                points.extend((x, y))
            if len(points) >= 4:
                self.chart.create_line(*points, fill=INK if not dashed else MUTED, dash=(5, 3) if dashed else (), width=1.5)
            stride = max(1, len(rows) // 45)
            for index in range(0, len(rows), stride):
                x, y = points[index * 2:index * 2 + 2]
                if dashed:
                    self.chart.create_rectangle(x - 2, y - 2, x + 2, y + 2, outline=MUTED)
                else:
                    self.chart.create_oval(x - 2, y - 2, x + 2, y + 2, fill=INK, outline=INK)
        self.chart.create_text(left, bottom + 12, text="1", anchor="w", fill=MUTED, font=self.small)
        self.chart.create_text(right, bottom + 12, text=display_number(count), anchor="e", fill=MUTED, font=self.small)
        self.chart.create_text(left + plot_width / 2, height - 9, text="completion order within each policy", fill=MUTED, font=self.small)

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.controller.stop()
        self.log("Closing safely. Waiting for active inference or background work to finish.")
        self.message.set("Closing safely. Active model execution or downloads must finish before the window closes.")
        self._render(self.controller.snapshot())

    def destroy(self):
        # Explicit destruction is also used by GUI tests. Remove callbacks so
        # Tcl cannot execute a now-deleted Python command in the next window.
        timer = getattr(self, "_poll_timer", None)
        if timer:
            try:
                self.after_cancel(timer)
            except tk.TclError:
                pass
        super().destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native deadline inference desktop app (no server).")
    parser.add_argument("--self-test", action="store_true", help="Open real Tk widgets and verify event-loop/layout startup, then close.")
    parser.add_argument("--smoke-inference", action="store_true", help="With --self-test, also run a short real comparison using cached weights.")
    parser.add_argument("--test-device", choices=("cpu", "cuda"), default="cpu", help=argparse.SUPPRESS)
    parser.add_argument("--test-model", default="mobilenet_v3_small", choices=[record["id"] for record in list_models()], help=argparse.SUPPRESS)
    parser.add_argument("--test-weights", default="", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help="JSON destination for the desktop self-test observations.")
    args = parser.parse_args(argv)
    if args.smoke_inference and not args.self_test:
        parser.error("--smoke-inference requires --self-test")
    if (args.test_model != "mobilenet_v3_small" or args.test_weights or args.test_device != "cpu") and not (args.self_test and args.smoke_inference):
        parser.error("--test-model/--test-weights/--test-device require --self-test --smoke-inference")
    try:
        app = DesktopApp()
    except tk.TclError as exc:
        raise SystemExit("A graphical desktop and Tk support are required. Install Python with Tcl/Tk (Linux: python3-tk). For headless use run the CLI demo instead. Details: " + str(exc)) from exc
    if args.self_test:
        app.self_test_passed = False
        started_at = time.monotonic()
        smoke_started = False
        def verify():
            nonlocal smoke_started
            elapsed = time.monotonic() - started_at
            if (app.checking or app.task_thread is not None) and elapsed < 60:
                app.after(100, verify)
                return
            if args.smoke_inference and not smoke_started and elapsed < 60:
                smoke_started = True
                app.model_label.set(app.model_labels[args.test_model])
                app._model_changed()
                for key, value in {"device": args.test_device, "duration_s": "0.3", "arrival_rate": "20", "pattern": "steady", "max_batch": "2", "deadline_ms": "1000", "policy": "compare", "model_id": args.test_model, "weights_path": args.test_weights}.items():
                    app.variables[key].set(value)
                if args.test_weights:
                    app.weights_source.set("Imported weights")
                app.run()
                app.after(100, verify)
                return
            state = app.controller.snapshot()
            if args.smoke_inference and state.get("status") in ACTIVE and elapsed < 60:
                app.after(100, verify)
                return
            # The worker may complete between normal 100 ms UI polls. Render
            # that same terminal snapshot before checking visible button state.
            app._render(state)
            app.update_idletasks()
            run_enabled = str(app.run_button["state"]) == "normal"
            cancel_disabled = str(app.cancel_button["state"]) == "disabled"
            observations = {"native_toolkit": "tkinter", "tk_version": str(app.tk.call("info", "patchlevel")),
                            "window_width": app.winfo_width(), "window_height": app.winfo_height(),
                            "config_fields": list(app.variables), "table_rows": len(app.results_table.get_children()),
                            "chart_width": app.chart.winfo_width(), "chart_height": app.chart.winfo_height(),
                            "run_enabled": run_enabled,
                            "cancel_disabled": cancel_disabled,
                            "server_started": False, "runtime_probe": app.runtime,
                            "inference_tested": args.smoke_inference, "status": state.get("status"),
                            "tested_model": args.test_model, "tested_device": args.test_device,
                            "model_backends": [r.get("backend", {}) for r in state.get("results", [])],
                            "error": state.get("error"), "results": [r.get("summary", {}) for r in state.get("results", [])],
                            "checks_passed": elapsed < 60 and bool(app.runtime) and run_enabled and cancel_disabled and set(app.variables) == set(asdict(Config())) and app.chart.winfo_width() > 100 and app.chart.winfo_height() > 60 and
                                             (not args.smoke_inference or state.get("status") == "completed" and len(state.get("results", [])) == 2 and
                                              all(r.get("backend", {}).get("device") == args.test_device and
                                                  r.get("summary", {}).get("completed_on_time", 0) + r.get("summary", {}).get("completed_late", 0) > 0 for r in state.get("results", [])))}
            text = json.dumps(observations, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text, encoding="utf-8")
            app.self_test_passed = observations["checks_passed"]
            import sys
            if sys.stdout is not None:
                print(text)
            app.close()
        app.after(450, verify)
    app.mainloop()
    if args.self_test and not app.self_test_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
