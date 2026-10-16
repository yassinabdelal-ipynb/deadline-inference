"""Inspect both release archives; optionally smoke-test an extracted Windows app.

No weights are downloaded. --smoke requires a deliberately populated cache via
DEADLINE_ENGINE_DATA_DIR (or the normal per-user cache).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all-models", action="store_true", help="With --smoke, also test all other classifier families using their cached checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT/"results"/"release-verification.json")
    args = parser.parse_args()
    manifest = ROOT/"dist"/"SHA256SUMS.txt"
    checks = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        path = manifest.parent/filename
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Checksum mismatch: {filename}")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if any(name.endswith((".pth", ".pt", ".onnx")) or any(part in (".venv", ".cache", ".build-venv") for part in Path(name).parts) for name in names):
                raise ValueError(f"Environment or model bulk included in {filename}")
            if filename.endswith("source.zip"):
                if any(name.endswith(("environment.json", "doctor-output.txt", "comparison.json", "requests.csv")) for name in names):
                    raise ValueError("Source archive contains raw local diagnostics or bulk traces")
            else:
                for required in ("DeadlineInference/DeadlineInference.exe", "DeadlineInference/THIRD_PARTY_LICENSES/manifest.json",
                                 "DeadlineInference/BUILD_INFO.json", "DeadlineInference/THIRD_PARTY_LICENSES/Tcl-9.0.4-license.terms",
                                 "DeadlineInference/THIRD_PARTY_LICENSES/Tk-9.0.4-license.terms", "DeadlineInference/THIRD_PARTY_LICENSES/OpenSSL-3.5.7-LICENSE.txt"):
                    if required not in names:
                        raise ValueError("Release missing " + required)
            checks.append({"archive": filename, "sha256": digest, "file_count": len(names), "bytes": path.stat().st_size})
    result = {"checks_passed": True, "archives": checks, "packaged_smoke": None,
              "scope": "Local Windows host only; not a second-PC or other-OS certification"}
    if args.all_models and not args.smoke:
        parser.error("--all-models requires --smoke")
    if args.smoke:
        if os.name != "nt":
            parser.error("The executable smoke test requires Windows.")
        scratch = ROOT/".cache"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="release QA with spaces ", dir=scratch) as folder:
            folder = Path(folder).resolve()
            if not folder.is_relative_to(scratch.resolve()):
                raise ValueError("Test extraction must remain inside the project cache")
            with zipfile.ZipFile(ROOT/"dist"/"DeadlineInference-Windows-x64-CPU.zip") as archive:
                for name in archive.namelist():
                    if not (folder/name).resolve().is_relative_to(folder):
                        raise ValueError("Unsafe archive path")
                archive.extractall(folder)
            observation = folder/"packaged-result.json"
            completed = subprocess.run([str(folder/"DeadlineInference"/"DeadlineInference.exe"), "--self-test", "--smoke-inference", "--output", str(observation)],
                                       capture_output=True, text=True, timeout=90, cwd=folder)
            if completed.returncode:
                raise RuntimeError(f"Extracted app failed ({completed.returncode}): {completed.stderr}")
            report = json.loads(observation.read_text(encoding="utf-8"))
            if not report.get("checks_passed") or not report.get("run_enabled") or not report.get("cancel_disabled"):
                raise RuntimeError("Extracted app smoke checks failed: " + json.dumps(report))
            result["packaged_smoke"] = report
            if args.all_models:
                sys.path.insert(0, str(ROOT))
                from deadline_engine.models import list_models
                result["additional_model_smokes"] = []
                for spec in list_models()[1:]:
                    # Deliberately supplied local checkpoint: inference never
                    # downloads a model just to make this test pass.
                    checkpoint = ROOT/".cache"/"torch"/"hub"/"checkpoints"/spec["filename"]
                    if not checkpoint.is_file():
                        raise ValueError(f"Explicitly fetch {spec['id']} before --all-models")
                    observation = folder/(spec["id"]+"-result.json")
                    completed = subprocess.run([str(folder/"DeadlineInference"/"DeadlineInference.exe"),
                                                "--self-test", "--smoke-inference", "--test-model", spec["id"],
                                                "--test-weights", str(checkpoint), "--output", str(observation)],
                                               capture_output=True, text=True, timeout=90, cwd=folder)
                    if completed.returncode:
                        raise RuntimeError(f"Extracted {spec['id']} failed ({completed.returncode}): {completed.stderr}")
                    report = json.loads(observation.read_text(encoding="utf-8"))
                    if not report.get("checks_passed") or report.get("tested_model") != spec["id"]:
                        raise RuntimeError(f"Extracted {spec['id']} smoke checks failed")
                    if any(info.get("model") != spec["id"] or info.get("weights_sha256") != spec["sha256"] for info in report["model_backends"]):
                        raise RuntimeError("Extracted model identity differs from its pinned checkpoint")
                    result["additional_model_smokes"].append(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Verified {len(checks)} archives. Extracted app tested: {args.smoke}. Evidence: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(str(exc))
