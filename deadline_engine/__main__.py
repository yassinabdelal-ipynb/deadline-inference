import argparse
from dataclasses import asdict
import json
from pathlib import Path
import threading

from .core import Config


def main():
    parser = argparse.ArgumentParser(description="Local image-classification inference and scheduling experiments")
    sub = parser.add_subparsers(dest="command")
    desktop = sub.add_parser("desktop", help="Open the native desktop application")
    desktop.add_argument("--self-test", action="store_true")
    desktop.add_argument("--smoke-inference", action="store_true")
    desktop.add_argument("--test-device", choices=("cpu", "cuda"), default="cpu", help=argparse.SUPPRESS)
    desktop.add_argument("--test-model", default="mobilenet_v3_small", help=argparse.SUPPRESS)
    desktop.add_argument("--test-weights", default="", help=argparse.SUPPRESS)
    desktop.add_argument("--output")
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--output")
    fetch = sub.add_parser("fetch-weights", help="Explicitly download a supported image classifier")
    fetch.add_argument("--model", default="mobilenet_v3_small")
    sub.add_parser("models", help="List supported image classifier families")
    imported = sub.add_parser("import-weights", help="Validate and cache a supported image classifier state_dict")
    imported.add_argument("path")
    imported.add_argument("--model", required=True)
    imported.add_argument("--classes", type=int, default=1000)
    imported.add_argument("--image-size", type=int, default=224)
    for name in ("demo", "benchmark", "profile"):
        p = sub.add_parser(name)
        p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        p.add_argument("--model", default="mobilenet_v3_small")
        p.add_argument("--weights", default="", help="Trusted .pth/.pt tensor state_dict; no executable models")
        p.add_argument("--classes", type=int, default=1000)
        p.add_argument("--image-size", type=int, default=224)
        p.add_argument("--duration", type=float, default=2 if name == "benchmark" else 4)
        p.add_argument("--rate", type=float, default=150 if name == "benchmark" else 100)
        p.add_argument("--deadline", type=float, default=100)
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--burst", type=float, default=5)
        p.add_argument("--queue", type=int, default=128)
        p.add_argument("--collection", type=float, default=4)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--pattern", choices=("steady", "bursty", "overload"), default="bursty")
        p.add_argument("--output", default="results/"+name)
        if name == "benchmark":
            p.add_argument("--repeats", type=int, default=3)
        if name == "profile":
            p.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if args.command in (None, "desktop"):
        try:
            from .desktop import main as desktop_main
        except ImportError as exc:
            if exc.name in ("tkinter", "_tkinter"):
                raise SystemExit("Native Tk support is missing. Install Python with Tcl/Tk (Linux: python3-tk), or use the CLI demo for headless operation.") from exc
            raise
        options = []
        if getattr(args, "self_test", False):
            options.append("--self-test")
        if getattr(args, "smoke_inference", False):
            options.append("--smoke-inference")
        if getattr(args, "test_device", "cpu") != "cpu":
            options.extend(("--test-device", args.test_device))
        if getattr(args, "output", None):
            options.extend(("--output", args.output))
        if getattr(args, "test_model", "mobilenet_v3_small") != "mobilenet_v3_small":
            options.extend(("--test-model", args.test_model))
        if getattr(args, "test_weights", ""):
            options.extend(("--test-weights", args.test_weights))
        desktop_main(options)
    elif args.command == "doctor":
        from .doctor import diagnose
        content = json.dumps(diagnose(), indent=2)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(content, encoding="utf-8")
        print(content)
    elif args.command == "fetch-weights":
        from .backend import fetch_weights
        print(json.dumps(fetch_weights(args.model), indent=2))
    elif args.command == "models":
        from .models import list_models
        print(json.dumps(list_models(), indent=2))
    elif args.command == "import-weights":
        from .backend import import_weights
        print(json.dumps(import_weights(args.path, args.model, args.classes, args.image_size), indent=2))
    else:
        from .experiment import aggregate, export_report, run_comparison
        config = Config(device=args.device, duration_s=args.duration, arrival_rate=args.rate,
                        deadline_ms=args.deadline, max_batch=args.batch, burst_intensity=args.burst,
                        queue_capacity=args.queue, collection_ms=args.collection, seed=args.seed, pattern=args.pattern,
                        model_id=args.model, weights_path=args.weights, num_classes=args.classes, image_size=args.image_size).validate()
        cancel = threading.Event()
        import signal
        signal.signal(signal.SIGINT, lambda *_: cancel.set())
        if args.command == "profile":
            from .profiling import run_profile, export_profile
            report = run_profile(config, cancel=cancel, samples=args.samples)
            export_profile(report, args.output)
            for row in report["rows"]:
                print(f"batch={row['batch_size']} samples={row['sample_count']} "
                      f"forward_mean_ms={row['device_ms']['mean']} service_mean_ms={row['service_ms']['mean']} "
                      f"images/s={row['sustained_images_per_second']}")
            print(f"Exports: {Path(args.output).resolve()}")
            if report["errors"]:
                raise RuntimeError("; ".join(report["errors"]))
            return
        if args.command == "demo":
            report = run_comparison(config, cancel=cancel)
        else:
            if not 1 <= args.repeats <= 10:
                parser.error("--repeats must be from 1 to 10")
            report = {"schema_version": 1, "config": asdict(config), "results": [],
                      "benchmark": {"repeats": args.repeats, "patterns": ["steady", "bursty", "overload"],
                                    "order": "alternate FIFO/deadline order by repeat to reduce ordering bias"}}
            for pattern in report["benchmark"]["patterns"]:
                for repeat in range(args.repeats):
                    if cancel.is_set():
                        break
                    c = Config(**{**asdict(config), "pattern": pattern, "seed": config.seed + repeat})
                    print(f"{pattern} repeat {repeat+1}/{args.repeats}: warming and measuring", flush=True)
                    order = ["fifo", "deadline"] if repeat % 2 == 0 else ["deadline", "fifo"]
                    run = run_comparison(c, cancel=cancel, order=order)
                    for r in run["results"]:
                        r["repeat"] = repeat
                    report["results"].extend(run["results"])
            report["aggregate"] = aggregate(report["results"])
        export_report(report, args.output)
        for result in report["results"]:
            s = result["summary"]
            print(f"{result['config']['pattern']} {result['policy']}: submitted={s['submitted']} "
                  f"on_time={s['completed_on_time']} late={s['completed_late']} expired={s['expired']} "
                  f"rejected={s['rejected']} throughput={s['throughput_rps']:.2f}/s p95={s['latency_ms']['p95']}")
        print(f"Exports: {Path(args.output).resolve()}")
        if any(r["errors"] for r in report["results"]):
            raise RuntimeError("An inference worker failed; inspect errors in comparison.json")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc))
