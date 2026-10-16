"""Validate and summarize an existing benchmark; never run inference.

Uses only Python's standard library. Defaults are anchored to the project so
the command also works when launched from a directory containing spaces.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
TERMINAL = (
    "completed_on_time", "completed_late", "rejected", "expired",
    "cancelled_queued", "cancelled_in_flight", "failed",
)
UNSUBMITTED = "cancelled_before_arrival"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, *, minimum=0):
    require(
        not isinstance(value, bool) and isinstance(value, (int, float))
        and math.isfinite(value) and value >= minimum,
        f"{label}: expected a finite number >= {minimum}, got {value!r}",
    )
    return value


def integer(value, label, *, minimum=0):
    number(value, label, minimum=minimum)
    require(isinstance(value, int), f"{label}: expected an integer")
    return value


def close(actual, expected, label):
    if expected is None:
        require(actual is None, f"{label}: expected null")
    else:
        number(actual, label)
        require(
            math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-5),
            f"{label}: reported {actual!r}, recomputed {expected!r}",
        )


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def spread(values):
    values = [value for value in values if value is not None]
    return {
        "sample_count": len(values),
        "mean": statistics.mean(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def canonical_plan(row, config, backend):
    """The application generates one fixed deadline/shape/dtype schedule.

    Cancelled-before-arrival rows retain their planned arrival and input index;
    metadata omitted from those rows has the generator's documented defaults.
    """
    return {
        "id": row["id"], "offset_s": row["arrival_s"],
        "deadline_ms": config["deadline_ms"], "priority": row.get("priority", 0),
        "input_index": row["input_index"],
        "shape": row.get("shape", backend.get("input_shape", [3, 224, 224])),
        "dtype": row.get("dtype", backend.get("input_dtype", "float32")),
    }


def validate_run(run, index):
    label = f"run {index} ({run.get('config', {}).get('pattern')}/{run.get('policy')})"
    config, summary = run["config"], run["summary"]
    trace, batches, backend = run["trace"], run["batches"], run["backend"]
    policy = run["policy"]
    require(policy in {"fifo", "deadline"}, f"{label}: invalid policy")
    maximum = integer(config["max_batch"], label + " max_batch", minimum=1)
    capacity = integer(config["queue_capacity"], label + " queue_capacity", minimum=1)
    integer(run.get("repeat", 0), label + " repeat")
    require(isinstance(trace, list) and isinstance(batches, list), f"{label}: traces/batches must be lists")
    integer(summary["planned"], label + " planned")
    require(len(trace) == summary["planned"], f"{label}: planned count differs from trace length")
    ids = [integer(row["id"], label + " request id") for row in trace]
    require(len(ids) == len(set(ids)), f"{label}: duplicate request IDs")
    require(set(ids) == set(range(len(trace))), f"{label}: generated planned IDs must exactly cover 0..planned-1")
    require(ids == sorted(ids), f"{label}: trace is not in planned request order")
    counts, members = Counter(), defaultdict(list)
    latencies, lateness, submitted_rows = [], [], []

    for row in trace:
        item = f"{label}, request {row['id']}"
        require(isinstance(row["submitted"], bool), item + ": submitted must be boolean")
        require(isinstance(row["admitted"], bool), item + ": admitted must be boolean")
        require(isinstance(row["executed"], bool), item + ": executed must be boolean")
        integer(row["input_index"], item + " input_index")
        number(row["arrival_s"], item + " arrival_s")
        status = row["status"]
        counts[status] += 1
        if not row["submitted"]:
            require(status == UNSUBMITTED, item + ": unsubmitted row lacks explicit cancellation")
            require(not row["admitted"] and not row["executed"], item + ": unsubmitted request admitted/executed")
            require(row.get("latency_ms") is None and row.get("finish_s") is None, item + ": unsubmitted row has completion timing")
            continue
        submitted_rows.append(row)
        require(status in TERMINAL, item + f": unknown/nonterminal status {status!r}")
        for field in ("deadline_s", "submission_s", "finish_s", "latency_ms", "generator_lateness_ms"):
            number(row[field], item + " " + field)
        require(row["submission_s"] + 1e-8 >= row["arrival_s"], item + ": submitted before planned arrival")
        require(row["finish_s"] + 1e-8 >= row["submission_s"], item + ": finish precedes submission")
        close(row["latency_ms"], (row["finish_s"] - row["arrival_s"]) * 1000, item + " latency_ms")
        close(row["generator_lateness_ms"], max(0, row["submission_s"] - row["arrival_s"]) * 1000, item + " generator_lateness_ms")
        close(row["deadline_s"] - row["arrival_s"], config["deadline_ms"] / 1000, item + " deadline interval")
        lateness.append(row["generator_lateness_ms"])
        if status.startswith("completed_"):
            require(row["admitted"] and row["executed"], item + ": completed request was not admitted/executed")
            latencies.append(row["latency_ms"])
            if status == "completed_on_time":
                require(row["finish_s"] <= row["deadline_s"] + 1e-8, item + ": on-time completion missed deadline")
            else:
                require(row["finish_s"] > row["deadline_s"] - 1e-8, item + ": late completion precedes deadline")
        if status == "rejected":
            require(not row["admitted"] and not row["executed"], item + ": rejected request admitted/executed")
        if status == "expired":
            require(not row["executed"], item + ": expired request was executed")
            require(row["finish_s"] + 1e-8 >= row["deadline_s"], item + ": expired before deadline")
        if status == "cancelled_queued":
            require(row["admitted"] and not row["executed"], item + ": invalid queued cancellation")
        if status == "cancelled_in_flight":
            require(row["admitted"] and row["executed"], item + ": invalid in-flight cancellation")
        if row["executed"]:
            require(row["admitted"], item + ": executed request was never admitted")
            integer(row.get("batch_id"), item + " batch_id")
            members[row["batch_id"]].append(row)
        else:
            require(row.get("batch_id") is None, item + ": unexecuted request has batch membership")

    planned = [canonical_plan(row, config, backend) for row in trace]
    require(all(a["offset_s"] <= b["offset_s"] for a, b in zip(planned, planned[1:])), label + ": unsorted planned arrivals")
    recomputed_hash = hashlib.sha256(json.dumps(planned, sort_keys=True).encode()).hexdigest()
    require(run["schedule_hash"] == recomputed_hash, label + ": stored schedule hash differs from reconstructed planned inputs")

    recorded_ids, overruns, ratios = set(), 0, []
    batch_sizes = []
    for batch in batches:
        batch_id = integer(batch["batch_id"], label + " batch_id")
        require(batch_id not in recorded_ids, label + ": duplicate batch ID")
        recorded_ids.add(batch_id)
        size = integer(batch["size"], label + " batch size", minimum=1)
        require(size <= maximum and (policy != "fifo" or size == 1), label + ": batch exceeds policy size limit")
        require(len(members[batch_id]) == size, label + f": batch {batch_id} membership differs from measured size")
        compatibility = {(tuple(row["shape"]), row["dtype"]) for row in members[batch_id]}
        require(len(compatibility) == 1, label + ": incompatible shape/dtype batched together")
        status = batch.get("status", "completed")  # Older successful reports omit status.
        require(status in {"completed", "failed"}, label + ": unknown batch status")
        for field in ("start_s", "finish_s", "service_ms"):
            number(batch[field], label + " batch " + field)
        if status == "failed":
            require(all(row["status"] == "failed" for row in members[batch_id]), label + ": failed batch has nonfailed requests")
            require(batch.get("device_ms") is None, label + ": failed batch must leave unknown device time null")
            require(isinstance(batch.get("error"), str) and batch["error"], label + ": failed batch lacks its error")
        else:
            number(batch["device_ms"], label + " batch device_ms")
        require(batch["finish_s"] >= batch["start_s"], label + ": negative batch wall time")
        require(batch["service_ms"] <= (batch["finish_s"] - batch["start_s"]) * 1000 + 1e-4, label + ": service time exceeds batch wall time")
        if batch["device_ms"] is not None:
            require(batch["device_ms"] <= batch["service_ms"] + 0.1, label + ": device time exceeds host service time")
        for row in members[batch_id]:
            close(row["finish_s"], batch["finish_s"], label + " batch/request finish")
            require(row["submission_s"] <= batch["start_s"] + 1e-8, label + ": batch began before request admission")
        estimate = backend.get("calibration", {}).get(str(size), {}).get("estimate_s")
        if estimate is not None and status == "completed":
            number(estimate, label + " runtime estimate", minimum=1e-12)
            ratio = batch["service_ms"] / (estimate * 1000)
            ratios.append(ratio)
            overruns += ratio > 1
        batch_sizes.append(size)

    failed_attempts = []
    for batch_id, group in members.items():
        require(len(group) <= maximum and (policy != "fifo" or len(group) == 1), label + ": attempted batch exceeds limit")
        if batch_id not in recorded_ids:
            require(all(row["status"] == "failed" for row in group), label + ": executed requests lack a measured batch or explicit failure")
            failed_attempts.append(batch_id)

    submitted = len(submitted_rows)
    completed = counts["completed_on_time"] + counts["completed_late"]
    expected = {
        "submitted": submitted, "admitted": sum(row["admitted"] for row in trace),
        "planned": len(trace), "unsubmitted_cancelled": counts[UNSUBMITTED],
        "completed": completed, "cancelled": counts["cancelled_queued"] + counts["cancelled_in_flight"],
        "deadline_misses": counts["expired"] + counts["completed_late"],
        "discarded": counts["rejected"] + counts["expired"],
        "batch_count": len(batches), "latency_sample_count": completed,
        **{status: counts[status] for status in TERMINAL},
    }
    for field, value in expected.items():
        integer(summary[field], label + " summary " + field)
        require(summary[field] == value, label + f": summary {field}={summary[field]} differs from trace {value}")
    require(sum(counts[status] for status in TERMINAL) == submitted, label + ": submitted terminal conservation failed")
    require(submitted + counts[UNSUBMITTED] == len(trace), label + ": planned cancellation conservation failed")
    for field in ("queue_depth", "in_flight"):
        require(summary[field] == 0, label + f": final {field} is nonzero")
    peak = integer(summary["peak_queue_depth"], label + " peak queue depth")
    require(peak <= capacity, label + ": peak queue exceeds capacity")
    for point in run.get("timeline", []):
        depth = integer(point["queue_depth"], label + " timeline queue depth")
        inflight = integer(point["in_flight"], label + " timeline in-flight size")
        require(depth <= peak <= capacity, label + ": timeline queue exceeds reported bound")
        require(inflight <= maximum and (policy != "fifo" or inflight <= 1), label + ": timeline in-flight limit exceeded")
    elapsed = number(summary["elapsed_s"], label + " elapsed", minimum=1e-12)
    close(summary["throughput_rps"], completed / elapsed, label + " throughput")
    close(summary["completion_rate"], completed / submitted if submitted else 0, label + " completion rate")
    close(summary["deadline_success_fraction"], counts["completed_on_time"] / submitted if submitted else 0, label + " deadline success")
    for name, quantile in (("p50", .5), ("p95", .95), ("p99", .99)):
        close(summary["latency_ms"][name], percentile(latencies, quantile), label + " " + name)
    close(summary["generator_lateness_ms"]["p95"], percentile(lateness, .95), label + " lateness p95")
    close(summary["generator_lateness_ms"]["max"], max(lateness, default=0), label + " lateness max")
    close(summary["device_ms_total"], sum(batch["device_ms"] for batch in batches if batch["device_ms"] is not None), label + " known device total")
    close(summary["service_ms_total"], sum(batch["service_ms"] for batch in batches), label + " service total")
    failed_batch_count = sum(batch.get("status") == "failed" for batch in batches)
    if "failed_batch_count" in summary:
        require(summary["failed_batch_count"] == failed_batch_count, label + ": failed batch count differs")
    if "device_timing_complete" in summary:
        require(summary["device_timing_complete"] == all(batch["device_ms"] is not None for batch in batches), label + ": device timing completeness differs")
    diagnostics = {
        "executed_requests": sum(row["executed"] for row in trace),
        "executed_failed_requests": sum(row["executed"] and row["status"] == "failed" for row in trace),
        "executed_cancelled_requests": counts["cancelled_in_flight"],
        "attempted_failed_batches_without_timing": len(failed_attempts),
        "recorded_failed_batches": failed_batch_count,
        "estimate_comparison_batch_count": len(ratios),
        "estimate_overrun_batch_count": overruns,
        "estimate_overrun_fraction": overruns / len(ratios) if ratios else None,
        "max_service_to_estimate_ratio": max(ratios, default=None),
        "batch_size_mean": statistics.mean(batch_sizes) if batch_sizes else None,
        "batch_size_max": max(batch_sizes, default=0),
    }
    return planned, {
        "pattern": config["pattern"], "policy": policy, "repeat": run.get("repeat", 0),
        "schedule_hash": run["schedule_hash"], "config": config,
        "summary": summary, "diagnostics": diagnostics, "errors": run.get("errors", []),
    }


def analyze(report):
    require(report.get("schema_version") in (1, 2), "Only report schema versions 1 and 2 are supported")
    results = report["results"]
    require(isinstance(results, list) and results, "Report contains no results")
    pairs, compact_runs = defaultdict(dict), []
    for index, run in enumerate(results):
        plan, compact = validate_run(run, index)
        key = (compact["pattern"], compact["repeat"])
        require(compact["policy"] not in pairs[key], f"Duplicate policy in pattern/repeat pair {key}")
        pairs[key][compact["policy"]] = (run, plan)
        compact_runs.append(compact)
    for key, pair in pairs.items():
        require(set(pair) == {"fifo", "deadline"}, f"Incomplete FIFO/deadline comparison pair {key}")
        first, second = pair["fifo"], pair["deadline"]
        require(first[0]["schedule_hash"] == second[0]["schedule_hash"], f"Pair {key} used different schedule hashes")
        require(first[1] == second[1], f"Pair {key} used different planned IDs, inputs, arrivals or deadlines")
        left = {k: v for k, v in first[0]["config"].items() if k != "policy"}
        right = {k: v for k, v in second[0]["config"].items() if k != "policy"}
        require(left == right, f"Pair {key} used different experiment configuration")
        for field in ("device", "device_name", "torch_version", "torchvision_version", "weights_sha256", "input_seed", "input_shape", "input_dtype"):
            require(first[0]["backend"].get(field) == second[0]["backend"].get(field), f"Pair {key} backend differs for {field}")

    groups = defaultdict(list)
    for run in compact_runs:
        groups[(run["pattern"], run["policy"])].append(run)
    aggregate = []
    for (pattern, policy), runs in groups.items():
        metrics = {}
        for name in ("throughput_rps", "deadline_success_fraction", "completion_rate", "discarded", "deadline_misses", "completed_late", "expired", "rejected"):
            metrics[name] = spread([run["summary"][name] for run in runs])
        for quantile in ("p50", "p95", "p99"):
            metrics[quantile + "_latency_ms"] = spread([run["summary"]["latency_ms"][quantile] for run in runs])
        metrics["generator_lateness_p95_ms"] = spread([run["summary"]["generator_lateness_ms"]["p95"] for run in runs])
        metrics["generator_lateness_max_ms"] = spread([run["summary"]["generator_lateness_ms"]["max"] for run in runs])
        for name in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "process_peak_resident_bytes"):
            metrics[name] = spread([run["summary"]["peak_memory"].get(name) for run in runs])
        compared = sum(run["diagnostics"]["estimate_comparison_batch_count"] for run in runs)
        overruns = sum(run["diagnostics"]["estimate_overrun_batch_count"] for run in runs)
        aggregate.append({
            "pattern": pattern, "policy": policy, "repeat_count": len(runs),
            "submitted_total": sum(run["summary"]["submitted"] for run in runs),
            "completed_total": sum(run["summary"]["completed"] for run in runs),
            "terminal_totals": {status: sum(run["summary"][status] for run in runs) for status in TERMINAL},
            "metrics": metrics,
            "estimate_overrun_batch_count": overruns,
            "estimate_comparison_batch_count": compared,
            "estimate_overrun_fraction": overruns / compared if compared else None,
            "max_service_to_estimate_ratio": max((run["diagnostics"]["max_service_to_estimate_ratio"] for run in runs if run["diagnostics"]["max_service_to_estimate_ratio"] is not None), default=None),
        })
    return {
        "schema_version": 1, "analysis_version": 1,
        "validation": {
            "passed": True, "run_count": len(results), "paired_comparison_count": len(pairs),
            "checks": ["exact planned request IDs and reconstructed schedule hash", "submitted/planned/admitted terminal conservation including cancellations", "paired identical planned arrivals, deadlines and input indices", "queue bounds and compatible batches; FIFO singleton", "executed request batch membership including explicit failed attempts", "reported timing, latency percentiles, throughput and counts recomputed"],
        },
        "benchmark": report.get("benchmark"), "config": report.get("config"),
        "backend": results[0]["backend"],
        "totals": {
            "planned": sum(run["summary"]["planned"] for run in compact_runs),
            "submitted": sum(run["summary"]["submitted"] for run in compact_runs),
            "completed": sum(run["summary"]["completed"] for run in compact_runs),
            "unsubmitted_cancelled": sum(run["summary"]["unsubmitted_cancelled"] for run in compact_runs),
            "executed": sum(run["diagnostics"]["executed_requests"] for run in compact_runs),
            **{status: sum(run["summary"][status] for run in compact_runs) for status in TERMINAL},
        },
        "aggregate": aggregate, "runs": compact_runs,
        "interpretation_limits": [
            "Three short repeats, when present, describe this experiment rather than sustained production capacity.",
            "Latency quantiles include completed requests only; deadline success uses all submitted requests, including rejected and expired work.",
            "Aggregate quantiles are means of per-run quantiles, not pooled request quantiles. Stdev is sample standard deviation across repeats.",
            "Runtime estimates are conservative calibration heuristics, not deadline guarantees. Overruns are observed host service durations above the stored estimate.",
            "Observed variation and generator lateness do not by themselves identify a cause; Windows scheduling, GPU clocks, thermal state and other host activity were not controlled.",
            "CUDA reserved memory includes cached warmup blocks; CPU peak is a process-lifetime high-water mark including preparation and earlier runs.",
            "Synthetic inputs do not measure model accuracy. Compare completion, deadline success and discarded work together.",
        ],
    }


def fmt(metric, scale=1, digits=1):
    if metric["mean"] is None:
        return "n/a"
    return f"{metric['mean'] * scale:.{digits}f} ± {metric['stdev'] * scale:.{digits}f}"


def markdown_report(result):
    backend, totals = result["backend"], result["totals"]
    lines = [
        "# Observed benchmark results", "",
        "Generated by `scripts/analyze_results.py` from the recorded comparison JSON. This command validates and summarizes existing data; it does not run inference.", "",
        f"Device: **{backend.get('device_name', backend.get('device'))}**; backend `{backend.get('torch_version')}`, torchvision `{backend.get('torchvision_version')}`, CUDA build `{backend.get('torch_cuda_version')}`. Model `{backend.get('model')}` with `{backend.get('weights')}`.", "",
        f"Validated {result['validation']['run_count']} runs / {result['validation']['paired_comparison_count']} paired schedules. Across all replays: {totals['submitted']:,} submitted, {totals['completed']:,} completed, {totals['expired']:,} expired, {totals['rejected']:,} rejected, {totals['failed']:,} failed; {totals['cancelled_queued'] + totals['cancelled_in_flight']:,} submitted cancellations and {totals['unsubmitted_cancelled']:,} cancellations before arrival. Every planned and submitted request is accounted for.", "",
        "Values below are mean ± sample standard deviation across repeats. Deadline success divides on-time completions by all submitted requests. Latency includes completed requests only.", "",
        "| Pattern | Policy | Repeats | Submitted total | Completed/s | Deadline success % | Completion % | Discarded/run | Late/run |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in result["aggregate"]:
        m = group["metrics"]
        lines.append(f"| {group['pattern']} | {group['policy']} | {group['repeat_count']} | {group['submitted_total']} | {fmt(m['throughput_rps'])} | {fmt(m['deadline_success_fraction'], 100)} | {fmt(m['completion_rate'], 100)} | {fmt(m['discarded'])} | {fmt(m['completed_late'])} |")
    lines.extend([
        "", "Discarded means rejected + expired before inference. Late completions performed model work and returned after the deadline; they are shown separately.", "",
        "| Pattern | Policy | p50 ms | p95 ms | p99 ms | Generator lateness p95 ms | Host estimate overruns/batches |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for group in result["aggregate"]:
        m = group["metrics"]
        lines.append(f"| {group['pattern']} | {group['policy']} | {fmt(m['p50_latency_ms'])} | {fmt(m['p95_latency_ms'])} | {fmt(m['p99_latency_ms'])} | {fmt(m['generator_lateness_p95_ms'])} | {group['estimate_overrun_batch_count']}/{group['estimate_comparison_batch_count']} |")
    lines.extend(["", "## Variability observed in these runs", ""])
    for group in result["aggregate"]:
        p95 = group["metrics"]["p95_latency_ms"]
        late = group["metrics"]["generator_lateness_max_ms"]
        ratio = group["max_service_to_estimate_ratio"]
        if p95["min"] is None:
            continue
        ratio_text = f"; largest service/estimate ratio {ratio:.2f}×" if ratio is not None else ""
        lines.append(f"- {group['pattern']}/{group['policy']}: per-run p95 {p95['min']:.1f}–{p95['max']:.1f} ms; largest observed generator lateness {late['max']:.1f} ms{ratio_text}.")
    burst = next((g for g in result["aggregate"] if g["pattern"] == "bursty" and g["policy"] == "deadline"), None)
    if burst:
        p95 = burst["metrics"]["p95_latency_ms"]
        if p95["min"] is not None:
            lines.extend(["", f"The deadline-aware bursty runs vary from {p95['min']:.1f} to {p95['max']:.1f} ms at p95. A larger throughput or deadline-success result does not imply lower completed-request latency in every repeat. Expiration changes which requests reach the latency sample; the completion and discard columns must be read together. Service-estimate overruns and generator lateness are measured observations, not proof of the cause of this variation."])
    lines.extend(["", "## Memory", "", "| Pattern | Policy | Max CUDA allocated MiB | Max CUDA reserved MiB | Max process peak MiB |", "|---|---|---:|---:|---:|"])
    for group in result["aggregate"]:
        values = [group["metrics"][name]["max"] for name in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "process_peak_resident_bytes")]
        rendered = [f"{value / 1024**2:.1f}" if value is not None else "n/a" for value in values]
        lines.append(f"| {group['pattern']} | {group['policy']} | {' | '.join(rendered)} |")
    lines.extend(["", "## Limits", ""])
    lines.extend("- " + text for text in result["interpretation_limits"])
    lines.extend(["", "## Reproduce this analysis", "", "```powershell", "& '.\\.venv\\Scripts\\python.exe' '.\\scripts\\analyze_results.py'", "```", "", "Raw per-request/batch observations remain in `results/benchmark/comparison.json`; compact validated summaries and exact per-run counts are in `results/benchmark/summary.json`.", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "benchmark" / "comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "benchmark" / "summary.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs" / "RESULTS.md")
    parser.add_argument("--no-markdown", action="store_true", help="Do not write the Markdown report")
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        def reject_constant(value):
            raise ValueError(f"Non-finite JSON constant {value}")
        report = json.loads(raw, parse_constant=reject_constant)
        result = analyze(report)
        result["source"] = {"path": str(args.input.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
        rendered = markdown_report(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if not args.no_markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(rendered, encoding="utf-8")
        # Console output stays ASCII so default Windows code pages are safe.
        print("Validated {run_count} runs and {paired_comparison_count} identical-schedule pairs.".format(**result["validation"]))
        print("Pattern   Policy     N   Completed/s mean +/- sd   On-time % mean +/- sd   p95 ms mean +/- sd   Estimate overruns")
        for group in result["aggregate"]:
            m = group["metrics"]
            cells = [fmt(m["throughput_rps"]), fmt(m["deadline_success_fraction"], 100), fmt(m["p95_latency_ms"])]
            cells = [cell.replace("±", "+/-") for cell in cells]
            print(f"{group['pattern']:<9} {group['policy']:<10} {group['repeat_count']:<3} {cells[0]:<25} {cells[1]:<24} {cells[2]:<22} {group['estimate_overrun_batch_count']}/{group['estimate_comparison_batch_count']}")
        print(f"Totals: submitted={result['totals']['submitted']}, completed={result['totals']['completed']}, failed={result['totals']['failed']}.")
        print(f"Compact JSON: {args.output.resolve()}")
        if not args.no_markdown:
            print(f"Markdown: {args.markdown.resolve()}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
