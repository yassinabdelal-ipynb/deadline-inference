"""Bounded, warmed image-model measurements; no queue, graphics or invented score.

This measures this model and configuration on the selected device. It does not
measure general CPU/GPU computational power or model prediction accuracy.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import io
import json
import math
from pathlib import Path
import statistics
import threading
import time
from types import SimpleNamespace

from .core import percentile
from .errors import PreparationCancelled


def describe(values):
    return {"mean": statistics.mean(values) if values else None,
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            "p50": percentile(values, .5), "p95": percentile(values, .95),
            "min": min(values) if values else None, "max": max(values) if values else None}


def run_profile(config, cancel=None, on_update=None, *, samples=10, backend=None):
    """One caller owns inference. Stop between forwards; never interrupt kernels."""
    config.validate()
    if isinstance(samples, bool) or not isinstance(samples, int) or not 3 <= samples <= 100:
        raise ValueError("Profile samples must be an integer from 3 through 100.")
    if backend is None:
        from .backend import TorchBackend
        backend = TorchBackend(config.device, config.max_batch, config.seed,
                               model_id=config.model_id, weights_path=config.weights_path,
                               num_classes=config.num_classes, image_size=config.image_size)
    cancel = cancel if cancel is not None else threading.Event()
    report = {"kind": "device_profile", "schema_version": 2,
              "created_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
              "backend": {}, "rows": [], "errors": [], "observer_errors": [],
              "cancelled": False, "samples_per_batch": samples, "warmup_per_batch": 2,
              "notes": ["Image classification only; synthetic inputs do not measure prediction accuracy.",
                        "No arrival queue; throughput here is measured images divided by summed host service time.",
                        "Forward time is CUDA event time on CUDA and host forward time on CPU.",
                        "Host service includes assembly, transfers, forward, and result return.",
                        "Model loading and calibration are outside measured samples.",
                        "These are model-specific measurements, not a universal hardware score.",
                        "Process memory is a lifetime peak; CUDA peaks reset after each batch warmup."]}
    if cancel.is_set():
        report["cancelled"] = True
        return report
    started = time.perf_counter()
    try:
        if hasattr(backend, "set_cancel_event"):
            backend.set_cancel_event(cancel)
        backend.prepare()
    except PreparationCancelled:
        report["cancelled"] = True
        report["preparation_s"] = time.perf_counter() - started
        report["backend"] = backend.metadata()
        return report
    report["preparation_s"] = time.perf_counter() - started
    report["backend"] = backend.metadata()

    def update(size, sample):
        if on_update and not report["observer_errors"]:
            try:
                on_update({"profile_batch_size": size, "profile_sample": sample,
                           "profile_samples": samples}, backend.metadata())
            except Exception as exc:
                report["observer_errors"].append(str(exc))

    for size in dict.fromkeys((1, config.max_batch)):
        if cancel.is_set():
            break
        requests = [SimpleNamespace(input_index=i, shape=(3, config.image_size, config.image_size),
                                    dtype="float32") for i in range(size)]
        measured = []
        errors = []
        update(size, 0)
        try:
            for _ in range(2):
                if cancel.is_set():
                    break
                backend.infer(requests)
            if hasattr(backend, "reset_peak_memory"):
                backend.reset_peak_memory()
            for index in range(samples):
                if cancel.is_set():
                    break
                observation = backend.infer(requests)
                for key in ("device_ms", "service_ms"):
                    value = observation[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                        raise RuntimeError(f"Invalid positive {key} from inference backend.")
                if len(observation["predictions"]) != size:
                    raise RuntimeError("Profile prediction count does not match batch size.")
                measured.append({"sample": index + 1, "device_ms": observation["device_ms"],
                                 "service_ms": observation["service_ms"]})
                update(size, index + 1)
        except Exception as exc:
            errors.append(str(exc))
            report["errors"].append(f"Batch {size}: {exc}")
        service_total = sum(x["service_ms"] for x in measured)
        report["rows"].append({"batch_size": size, "sample_count": len(measured),
                               "device_ms": describe([x["device_ms"] for x in measured]),
                               "service_ms": describe([x["service_ms"] for x in measured]),
                               "images_per_second": describe([size * 1000 / x["service_ms"] for x in measured]),
                               "sustained_images_per_second": size * len(measured) * 1000 / service_total if service_total else None,
                               "peak_memory": backend.peak_memory(), "samples": measured, "errors": errors})
        if errors:
            break
    report["cancelled"] = cancel.is_set()
    return report


def profile_csv(report):
    stream = io.StringIO(newline="")
    fields = ["model", "weights_sha256", "device", "image_size", "num_classes", "batch_size", "sample_count",
              "forward_mean_ms", "forward_stdev_ms", "forward_p95_ms", "service_mean_ms", "service_stdev_ms",
              "service_p95_ms", "images_per_second", "images_per_second_stdev", "parameter_count",
              "model_parameter_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
              "process_peak_resident_bytes", "measurement_status", "cancelled", "errors"]
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    info, config = report.get("backend", {}), report["config"]
    no_samples = {"batch_size": None, "sample_count": 0, "device_ms": describe([]),
                  "service_ms": describe([]), "images_per_second": describe([]),
                  "sustained_images_per_second": None, "peak_memory": {}, "errors": report.get("errors", [])}
    for row in report["rows"] or [no_samples]:
        writer.writerow({"model": info.get("model", config.get("model_id")), "weights_sha256": info.get("weights_sha256"),
                         "device": config["device"], "image_size": config["image_size"], "num_classes": config["num_classes"],
                         "batch_size": row["batch_size"], "sample_count": row["sample_count"],
                         "forward_mean_ms": row["device_ms"]["mean"], "forward_stdev_ms": row["device_ms"]["stdev"],
                         "forward_p95_ms": row["device_ms"]["p95"], "service_mean_ms": row["service_ms"]["mean"],
                         "service_stdev_ms": row["service_ms"]["stdev"], "service_p95_ms": row["service_ms"]["p95"],
                         "images_per_second": row["sustained_images_per_second"],
                         "images_per_second_stdev": row["images_per_second"]["stdev"],
                         "parameter_count": info.get("parameter_count"), "model_parameter_bytes": info.get("model_parameter_bytes"),
                         **{key: row["peak_memory"].get(key) for key in fields if key.endswith("resident_bytes") or key.startswith("cuda_peak_")},
                         "measurement_status": "cancelled" if report["cancelled"] else "failed" if row["errors"] else "measured" if row["sample_count"] else "no_samples",
                         "cancelled": report["cancelled"], "errors": "; ".join(row["errors"])})
    return stream.getvalue()


def export_profile(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "profile.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output / "profile.csv").write_text(profile_csv(report), encoding="utf-8", newline="")
