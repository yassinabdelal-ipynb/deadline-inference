"""Export an allowlisted public measurement snapshot and two dependency-free SVGs.

Private raw validation traces are deliberately not needed to redraw the figures:
    python scripts/render_results.py --from-public docs/public/measurements.json
To replace the snapshot with a new completed local matrix:
    python scripts/render_results.py --source results/image-models/validation.json --environment results/image-model-environment.json
"""
import argparse
import csv
from collections import defaultdict
from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "resnet18": "ResNet18",
    "efficientnet_b0": "EfficientNet-B0",
    "squeezenet1_1": "SqueezeNet1.1",
}
TERMINALS = (
    "completed_on_time", "completed_late", "expired", "rejected",
    "failed", "cancelled_queued", "cancelled_in_flight",
)


def public_snapshot(raw, environment):
    """Allowlist numeric/identity fields; never copy local paths or diagnostics."""
    if not raw.get("checks_passed"):
        raise ValueError("The input matrix has not passed its validation checks.")
    profiles = []
    for profile in raw["profiles"]:
        backend = profile["backend"]
        profiles.append({
            "model": profile["model"], "device": profile["device"],
            "device_name": backend["device_name"],
            "torch_version": backend["torch_version"],
            "torchvision_version": backend["torchvision_version"],
            "torch_cuda_version": backend["torch_cuda_version"],
            "input_shape": backend["input_shape"],
            "input_dtype": backend["input_dtype"],
            "cpu_threads": backend["torch_cpu_threads"],
            "weights_sha256": backend["weights_sha256"],
            "parameter_count": backend["parameter_count"],
            "parameter_bytes": backend["model_parameter_bytes"],
            "rows": [{key: row[key] for key in (
                "batch_size", "sample_count", "device_ms", "service_ms",
                "sustained_images_per_second", "peak_memory", "samples",
            )} for row in profile["rows"]],
        })
    replays = []
    for replay in raw["comparisons"]:
        summary = replay["summary"]
        selected = {key: summary[key] for key in (
            "submitted", "admitted", *TERMINALS, "completed", "planned",
            "unsubmitted_cancelled", "elapsed_s", "throughput_rps",
            "completion_rate", "deadline_success_fraction", "discarded",
            "latency_ms", "latency_sample_count", "mean_batch_size",
            "generator_lateness_ms", "peak_queue_depth", "peak_memory",
        )}
        if selected["submitted"] != sum(selected[key] for key in TERMINALS):
            raise ValueError("A replay does not conserve submitted requests.")
        replays.append({**{key: replay[key] for key in (
            "model", "device", "repeat", "policy", "schedule_hash",
        )}, "summary": selected})
    gpu = next(iter(environment.get("torch", {}).get("devices", [])), {})
    inventory = environment.get("commands", {}).get("nvidia_smi", {}).get("stdout", "")
    driver = inventory.split(",")[1].strip() if inventory.count(",") >= 1 else None
    os_info = environment.get("os", {})
    return {
        "schema_version": 1,
        "scope": "Observed on one interactive Windows laptop; synthetic inputs do not measure accuracy.",
        "host": {
            "os": " ".join(str(os_info.get(key, "")) for key in ("system", "release", "version", "machine")).strip() or None,
            "python": environment.get("python", {}).get("version", "").split(" ")[0] or None,
            "gpu": gpu.get("name"),
            "vram_mib": gpu.get("total_vram_bytes", 0)/(1024**2) if gpu else None,
            "driver": driver,
            "cpu": environment.get("cpu", {}).get("processor"),
            "logical_processors": environment.get("cpu", {}).get("logical_processors"),
        },
        "workload": {
            "input": "procedural RGB float32 224 x 224; ImageNet normalization",
            "profile_batches": [1, 4], "samples_per_batch": 10,
            "timing": "Setup/calibration and two additional warmups excluded; CUDA events synchronized.",
            "scheduling_pattern": "bursty", "arrival_window_seconds": 0.6,
            "base_rate_per_second": 40, "burst_multiplier": 5,
            "deadline_ms": 1000, "max_batch": 4, "collection_ms": 4,
            "queue_capacity": 128, "seeds": [7, 8],
            "policy_order": "alternates between seeds",
            "arrival_clock": "Corrected monotonic wait loop; submissions cannot precede planned arrivals.",
        },
        "models": [{key: model[key] for key in ("model", "url", "sha256", "bytes")}
                   for model in raw["models"]],
        "profiles": profiles, "replays": replays,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(data):
    profiles = []
    for profile in data["profiles"]:
        for row in profile["rows"]:
            profiles.append({
                "model": profile["model"], "device": profile["device"],
                "batch_size": row["batch_size"], "samples": row["sample_count"],
                "service_mean_ms": row["service_ms"]["mean"],
                "service_stdev_ms": row["service_ms"]["stdev"],
                "forward_mean_ms": row["device_ms"]["mean"],
                "forward_stdev_ms": row["device_ms"]["stdev"],
                "images_per_second": row["sustained_images_per_second"],
                "cuda_peak_allocated_bytes": row["peak_memory"]["cuda_peak_allocated_bytes"],
                "cuda_peak_reserved_bytes": row["peak_memory"]["cuda_peak_reserved_bytes"],
                "process_lifetime_peak_bytes": row["peak_memory"]["process_peak_resident_bytes"],
                "weights_sha256": profile["weights_sha256"],
            })
    replays = []
    for replay in data["replays"]:
        summary = replay["summary"]
        row = {key: replay[key] for key in ("model", "device", "repeat", "policy", "schedule_hash")}
        row.update({key: summary[key] for key in (
            "submitted", "admitted", *TERMINALS, "completed", "elapsed_s", "throughput_rps",
            "completion_rate", "deadline_success_fraction", "discarded", "latency_sample_count",
            "mean_batch_size", "peak_queue_depth",
        )})
        row.update({f"latency_{key}_ms": value for key, value in summary["latency_ms"].items()})
        row["generator_lateness_p95_ms"] = summary["generator_lateness_ms"]["p95"]
        replays.append(row)
    return profiles, replays


def svg_start(title, description, height=690):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="{height}" viewBox="0 0 1040 {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
        '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse"><rect width="6" height="6" fill="#DDD8CD"/><path d="M0 6L6 0" stroke="#545A50" stroke-width="1"/></pattern></defs>',
        '<rect width="1040" height="100%" fill="#FFFDF8"/>',
        '<g font-family="system-ui, -apple-system, Segoe UI, sans-serif" fill="#242722">',
        f'<text x="28" y="37" font-size="22" font-weight="650">{escape(title)}</text>',
    ]


def throughput_svg(data):
    svg = svg_start("Measured image inference throughput", "Batch one and batch four images per second for four classifiers on CPU and CUDA; 10 samples per batch.")
    svg.append('<text x="28" y="64" font-size="13" fill="#545A50">224 x 224 float32 | 10 timed forwards per bar | setup and warmup excluded</text>')
    svg.extend([
        '<rect x="28" y="82" width="18" height="12" fill="url(#hatch)" stroke="#545A50"/><text x="54" y="93" font-size="13">Batch 1</text>',
        '<rect x="145" y="82" width="18" height="12" fill="#465D45"/><text x="171" y="93" font-size="13">Batch 4</text>',
    ])
    origin, width, maximum = 264, 680, 200
    for tick in range(0, 201, 50):
        x = origin + width * tick / maximum
        svg.append(f'<path d="M{x} 119V615" stroke="#DDD8CD"/><text x="{x}" y="637" text-anchor="middle" font-size="12">{tick}</text>')
    for index, profile in enumerate(data["profiles"]):
        y = 130 + index * 60
        label = f'{LABELS[profile["model"]]} / {profile["device"].upper()}'
        svg.append(f'<text x="244" y="{y+16}" text-anchor="end" font-size="13">{escape(label)}</text>')
        for j, row in enumerate(profile["rows"]):
            rate = row["sustained_images_per_second"]
            bar_y = y + j*21
            fill = "url(#hatch)" if row["batch_size"] == 1 else "#465D45"
            svg.append(f'<rect x="{origin}" y="{bar_y}" width="{rate/maximum*width:.2f}" height="16" fill="{fill}" stroke="#545A50" stroke-width="0.7"/>')
            svg.append(f'<text x="{origin+rate/maximum*width+7:.2f}" y="{bar_y+12}" font-size="12">{rate:.2f}</text>')
    svg.extend([
        '<text x="604" y="660" text-anchor="middle" font-size="13">images / second (measured images divided by total host service time)</text>',
        '<text x="28" y="682" font-size="12" fill="#545A50">One interactive Windows laptop. Substantial variance; see CSV/JSON for sample SD. Not a hardware score.</text>',
        '</g></svg>',
    ])
    return "\n".join(svg) + "\n"


def outcomes_svg(data):
    grouped = defaultdict(lambda: defaultdict(int))
    for replay in data["replays"]:
        for key in ("submitted", *TERMINALS):
            grouped[(replay["model"], replay["device"])][key] += replay["summary"][key]
    totals = {key: sum(replay["summary"][key] for replay in data["replays"]) for key in ("submitted", *TERMINALS)}
    svg = svg_start("Every submitted request accounted for", f'Outcomes combined across both policies and two seeds per model and device. {totals["submitted"]} submitted, {totals["completed_on_time"]} on time, {totals["completed_late"]} late, {totals["expired"]} expired. Not a policy comparison.', height=620)
    svg.append('<text x="28" y="64" font-size="13" fill="#545A50">32 corrected replays / 16 matched pairs | both policies combined here to show accounting</text>')
    for x, label, fill in ((28, "On time", "#465D45"), (180, "Completed late", "#C5BFB3"), (380, "Expired before inference", "url(#hatch)")):
        svg.append(f'<rect x="{x}" y="82" width="18" height="12" fill="{fill}" stroke="#545A50"/><text x="{x+26}" y="93" font-size="13">{label}</text>')
    origin, width = 264, 680
    for index, profile in enumerate(data["profiles"]):
        counts = grouped[(profile["model"], profile["device"])]
        total = counts["submitted"]
        y = 127 + index*50
        label = f'{LABELS[profile["model"]]} / {profile["device"].upper()}'
        svg.append(f'<text x="244" y="{y+19}" text-anchor="end" font-size="13">{escape(label)}</text>')
        offset = 0
        for key, fill, text_fill in (("completed_on_time", "#465D45", "#FFFDF8"), ("completed_late", "#C5BFB3", "#242722"), ("expired", "url(#hatch)", "#242722")):
            count = counts[key]
            segment = width*count/total
            svg.append(f'<rect x="{origin+offset:.2f}" y="{y}" width="{segment:.2f}" height="28" fill="{fill}" stroke="#FFFDF8" stroke-width="1"/>')
            if segment > 30:
                # Opaque label backing keeps hatched sections readable.
                if key == "expired":
                    svg.append(f'<rect x="{origin+offset+segment/2-17:.2f}" y="{y+5}" width="34" height="18" fill="#DDD8CD"/>')
                svg.append(f'<text x="{origin+offset+segment/2:.2f}" y="{y+19}" fill="{text_fill}" text-anchor="middle" font-size="12">{count}</text>')
            offset += segment
        svg.append(f'<text x="959" y="{y+19}" font-size="12">{total}</text>')
    svg.extend([
        f'<text x="28" y="557" font-size="17" font-weight="600">{totals["submitted"]:,} submitted = {totals["completed_on_time"]:,} on time + {totals["completed_late"]:,} late + {totals["expired"]:,} expired</text>',
        '<text x="28" y="583" font-size="12" fill="#545A50">Zero rejected, failed or cancelled in this fixture. Expiry is discarded work, not a successful answer.</text>',
        '<text x="28" y="605" font-size="12" fill="#545A50">Equal relative deadlines make EDF request order match FIFO; batching and collection also change between policies.</text>',
        '</g></svg>',
    ])
    return "\n".join(svg) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", type=Path)
    source.add_argument("--from-public", type=Path, default=None)
    parser.add_argument("--environment", type=Path, help="Optional matching doctor JSON; only hardware/version fields are exported.")
    parser.add_argument("--output", type=Path, default=ROOT/"docs"/"public")
    args = parser.parse_args()
    if args.source:
        environment = json.loads(args.environment.read_text(encoding="utf-8")) if args.environment else {}
        data = public_snapshot(json.loads(args.source.read_text(encoding="utf-8")), environment)
    else:
        public_path = args.from_public or ROOT/"docs"/"public"/"measurements.json"
        data = json.loads(public_path.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"measurements.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    profiles, replays = csv_rows(data)
    write_csv(args.output/"profiles.csv", profiles)
    write_csv(args.output/"replays.csv", replays)
    (args.output/"throughput.svg").write_text(throughput_svg(data), encoding="utf-8")
    (args.output/"outcomes.svg").write_text(outcomes_svg(data), encoding="utf-8")
    print(f"Exported {len(profiles)} profile rows and {len(replays)} replays to {args.output}.")


if __name__ == "__main__":
    main()
