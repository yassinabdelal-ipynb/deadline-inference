"""Actual public-checkpoint import, CPU/CUDA profiling and matched replay checks.

Downloads only with --fetch. Runs sequentially on this computer; results identify
the actual device and configuration. This is not an accuracy evaluation.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deadline_engine.backend import CHECKPOINTS, fetch_weights, import_weights
from deadline_engine.core import Config, TERMINAL
from deadline_engine.experiment import export_report, run_comparison
from deadline_engine.models import list_models
from deadline_engine.profiling import export_profile, run_profile
from analyze_results import analyze


def check_comparison(report):
    first, second = report["results"]
    assert first["schedule_hash"] == second["schedule_hash"], "Different arrival schedules"
    assert [r["input_index"] for r in first["trace"]] == [r["input_index"] for r in second["trace"]]
    for result in (first, second):
        summary = result["summary"]
        assert not result["errors"], result["errors"]
        assert summary["failed"] == 0
        assert summary["submitted"] > 0
        assert summary["submitted"] == sum(summary[key] for key in TERMINAL)
        assert summary["planned"] == summary["submitted"] + summary["unsubmitted_cancelled"]
        assert summary["peak_queue_depth"] <= result["config"]["queue_capacity"]
        assert all(r["shape"] == (3, result["config"]["image_size"], result["config"]["image_size"]) for r in result["trace"])
        assert all(batch["size"] <= (1 if result["policy"] == "fifo" else result["config"]["max_batch"]) for batch in result["batches"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Explicitly acquire official checkpoints before testing")
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu"])
    parser.add_argument("--models", nargs="+", default=[spec["id"] for spec in list_models()])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reuse-profiles", action="store_true", help="Reuse and verify saved device profiles; rerun all scheduling replays")
    parser.add_argument("--output", type=Path, default=ROOT/"results"/"image-models")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3:
        parser.error("--repeats must be 1 through 3")
    args.output.mkdir(parents=True, exist_ok=True)
    aggregate = {"checks_passed": False, "scope": "Actual local host only; synthetic images do not measure accuracy",
                 "models": [], "profiles": [], "comparisons": []}
    for model_id in args.models:
        spec = next((record for record in list_models() if record["id"] == model_id), None)
        if spec is None:
            parser.error(f"Unsupported model: {model_id}")
        acquired = fetch_weights(model_id) if args.fetch else {"path": str(CHECKPOINTS/spec["filename"])}
        print(f"Importing actual checkpoint: {model_id}", flush=True)
        imported = import_weights(acquired["path"], model_id)
        assert imported["sha256"] == spec["sha256"]
        aggregate["models"].append({"model": model_id, "url": spec["url"], "sha256": imported["sha256"],
                                    "bytes": Path(imported["path"]).stat().st_size, "import_validated": True})
        for device in args.devices:
            config = Config(model_id=model_id, weights_path=imported["path"], device=device,
                            max_batch=4, pattern="bursty", duration_s=.6, arrival_rate=40,
                            deadline_ms=1000, seed=7)
            folder = args.output/model_id/device
            print(f"Device test {model_id} / {device}", flush=True)
            if args.reuse_profiles:
                profile = json.loads((folder/"profile"/"profile.json").read_text(encoding="utf-8"))
                assert profile["config"] == asdict(config), "Saved profile has different settings"
                assert not profile["cancelled"]
            else:
                profile = run_profile(config, samples=10)
                export_profile(profile, folder/"profile")
            assert not profile["errors"], profile["errors"]
            assert len(profile["rows"]) == 2 and all(r["sample_count"] == 10 for r in profile["rows"])
            assert profile["backend"]["weights_sha256"] == imported["sha256"]
            assert profile["backend"]["device"] == device
            aggregate["profiles"].append({"model": model_id, "device": device,
                                          "backend": profile["backend"], "rows": profile["rows"]})
            combined = {"schema_version": 2, "config": asdict(config), "results": []}
            for repeat in range(args.repeats):
                repeat_config = replace(config, seed=7 + repeat)
                order = ["fifo", "deadline"] if repeat % 2 == 0 else ["deadline", "fifo"]
                print(f"Scheduling {model_id} / {device} repeat {repeat + 1}", flush=True)
                report = run_comparison(repeat_config, order=order)
                check_comparison(report)
                for result in report["results"]:
                    result["repeat"] = repeat
                    aggregate["comparisons"].append({"model": model_id, "device": device, "repeat": repeat,
                                                     "policy": result["policy"], "schedule_hash": result["schedule_hash"],
                                                     "summary": result["summary"]})
                combined["results"].extend(report["results"])
            export_report(combined, folder/"comparison")
            independently_checked = analyze(combined)
            (folder/"comparison"/"analysis.json").write_text(
                json.dumps(independently_checked, indent=2, allow_nan=False), encoding="utf-8")
    aggregate["checks_passed"] = True
    destination = args.output/"validation.json"
    destination.write_text(json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8")
    print(f"PASS: {len(aggregate['models'])} imported models, {len(aggregate['profiles'])} device profiles, "
          f"{len(aggregate['comparisons'])} scheduling replays. {destination}")


if __name__ == "__main__":
    main()
