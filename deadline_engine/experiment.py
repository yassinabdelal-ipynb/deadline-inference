"""Sequential image-model experiments and exports shared by CLI and native UI."""
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import io
import json
from pathlib import Path
import statistics

from .core import Config, make_schedule, replay


def run_comparison(config, cancel=None, on_update=None, on_result=None, order=None):
    from .backend import TorchBackend
    schedule = make_schedule(config)
    backend = TorchBackend(config.device, config.max_batch, config.seed,
                           model_id=config.model_id, weights_path=config.weights_path,
                           num_classes=config.num_classes, image_size=config.image_size)
    policies = order or (["fifo", "deadline"] if config.policy == "compare" else [config.policy])
    results = []
    for policy in policies:
        if cancel is not None and cancel.is_set():
            break
        def update(live):
            if on_update:
                on_update(policy, live, backend.metadata())
        result = replay(config, schedule, policy, backend, cancel=cancel, on_update=update)
        results.append(result)
        if on_result:
            on_result(result)
        if result["errors"]:
            break
    return {"schema_version": 2, "kind": "scheduling_comparison", "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config), "results": results,
            "notes": ["Synthetic images measure scheduling, not prediction accuracy.",
                      "Policies replay identical planned arrivals and input indices sequentially.",
                      "Latency quantiles include completed requests only; deadline success denominator is all submitted requests.",
                      "Cancelled-before-arrival requests are planned but unsubmitted; excluded from submitted denominator."]}


def summary_csv(report):
    stream = io.StringIO(newline="")
    fields = ["model", "weights_sha256", "device", "image_size", "num_classes", "pattern", "repeat", "policy", "schedule_hash", "submitted", "admitted", "completed_on_time",
              "completed_late", "expired", "rejected", "cancelled_queued", "cancelled_in_flight", "failed",
              "unsubmitted_cancelled", "throughput_rps", "completion_rate", "deadline_success_fraction",
              "deadline_misses", "discarded", "p50_ms", "p95_ms", "p99_ms", "peak_queue_depth",
              "batch_count", "mean_batch_size", "forward_ms_per_image", "queue_wait_p95_ms", "failed_batch_count", "device_timing_complete", "device_ms_total", "service_ms_total",
              "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "process_peak_resident_bytes"]
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for result in report["results"]:
        s = result["summary"]
        row = {key: s.get(key) for key in fields}
        backend = result.get("backend", {})
        row.update(model=backend.get("model"), weights_sha256=backend.get("weights_sha256"),
                   device=backend.get("device"), image_size=result["config"].get("image_size", 224),
                   num_classes=result["config"].get("num_classes", 1000),
                   queue_wait_p95_ms=s.get("queue_wait_ms", {}).get("p95"),
                   policy=result["policy"], schedule_hash=result["schedule_hash"],
                   pattern=result["config"]["pattern"], repeat=result.get("repeat", 0),
                   **{f"{q}_ms": s["latency_ms"][q] for q in ("p50", "p95", "p99")},
                   **{k: s["peak_memory"].get(k) for k in fields if k.endswith("_bytes")})
        writer.writerow(row)
    return stream.getvalue()


def export_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output / "comparison.csv").write_text(summary_csv(report), encoding="utf-8", newline="")
    with (output / "requests.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["policy", "pattern", "repeat", "id", "status", "submitted", "admitted", "arrival_s",
                  "submission_s", "deadline_s", "finish_s", "latency_ms", "generator_lateness_ms",
                  "input_index", "priority", "shape", "dtype", "batch_id", "executed", "prediction", "error"]
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        for result in report["results"]:
            for row in result["trace"]:
                writer.writerow({**row, "policy": result["policy"], "pattern": result["config"]["pattern"],
                                 "repeat": result.get("repeat", 0)})


def aggregate(results):
    groups = {}
    for result in results:
        groups.setdefault((result["config"]["pattern"], result["policy"]), []).append(result["summary"])
    output = []
    for (pattern, policy), rows in groups.items():
        metrics = {}
        for key in ("throughput_rps", "deadline_success_fraction", "completion_rate"):
            values = [r[key] for r in rows]
            metrics[key] = {"mean": statistics.mean(values), "stdev": statistics.stdev(values) if len(values)>1 else 0,
                            "min": min(values), "max": max(values)}
        p95 = [r["latency_ms"]["p95"] for r in rows if r["latency_ms"]["p95"] is not None]
        metrics["p95_latency_ms"] = {"mean": statistics.mean(p95) if p95 else None,
                                    "stdev": statistics.stdev(p95) if len(p95)>1 else 0}
        output.append({"pattern": pattern, "policy": policy, "repeat_count": len(rows),
                       "submitted_total": sum(r["submitted"] for r in rows), "metrics": metrics})
    return output
