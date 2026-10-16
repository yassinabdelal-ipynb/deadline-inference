"""Check publication inputs without reading credentials or contacting a service.

python scripts/check_release.py --source
python scripts/check_release.py --archives

Checks are a bounded publication guard, not a security audit or malware scan.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

try:
    from .build_release import ROOT, source_allowed, source_files
except ImportError:
    from build_release import ROOT, source_allowed, source_files

TEXT_SUFFIXES = {".py", ".ps1", ".cmd", ".md", ".txt", ".yml", ".yaml", ".json", ".toml", ".csv", ".svg", ".cff"}
FORBIDDEN_PARTS = {".git", ".venv", ".venv-cpu", ".venv-cuda", ".build-venv", ".cache", ".build", "__pycache__", "node_modules"}
FORBIDDEN_SUFFIXES = {".pth", ".pt", ".onnx", ".safetensors", ".pem", ".key", ".pfx", ".log"}
SECRET_PATTERNS = [
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]
PRIVATE_PATH = re.compile(r"(?:[A-Za-z]:[/\\]Users[/\\](?!Public\b|YourName\b|USERNAME\b)[^/\\\s'\"<>]+|/(?:home|Users)/(?!user(?:/|\b)|username\b)[^/\s'\"<>]+)")


def check_text(name, data):
    if Path(name).suffix.lower() not in TEXT_SUFFIXES and Path(name).name not in {"LICENSE", ".gitignore"}:
        return
    text = data.decode("utf-8", errors="replace")
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        raise ValueError(f"Potential credential in {name}; inspect it locally. Matched value is deliberately not printed.")
    if PRIVATE_PATH.search(text):
        raise ValueError(f"Personal home-directory path in {name}; replace it with a portable example.")


def check_source():
    files = source_files()
    for path in files:
        relative = path.relative_to(ROOT)
        if not source_allowed(relative):
            raise ValueError("Source allowlist mismatch: " + str(relative))
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Source link leaves the project: " + str(relative))
        check_text(str(relative), path.read_bytes())
    return {"files_checked": len(files), "scope": "explicit source archive allowlist"}


def safe_member(name):
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
        raise ValueError("Unsafe archive member: " + name)
    if any(part in FORBIDDEN_PARTS or part.startswith(".env") for part in path.parts):
        raise ValueError("Private environment in archive: " + name)
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError("Model, credential or generated bulk in archive: " + name)
    return path


def check_archive(path, source=False):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate archive member: " + path.name)
        for item in archive.infolist():
            member = safe_member(item.filename)
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlink in release: " + item.filename)
            if source:
                if member.parts[0] != "deadline-inference" or not source_allowed(Path(*member.parts[1:])):
                    raise ValueError("Source archive member outside allowlist: " + item.filename)
            elif member.parts[0] != "DeadlineInference":
                raise ValueError("Unexpected desktop archive root: " + item.filename)
            if item.is_dir():
                continue
            # Scan project text and metadata. Binary dependency contents are
            # third-party runtime files, not source and not credential evidence.
            if source or "_internal" not in member.parts and "THIRD_PARTY_LICENSES" not in member.parts:
                check_text(item.filename, archive.read(item))
        required = {"deadline-inference/README.md", "deadline-inference/LICENSE", "deadline-inference/deadline_engine/core.py"} if source else {
            "DeadlineInference/DeadlineInference.exe", "DeadlineInference/BUILD_INFO.json", "DeadlineInference/LICENSE",
            "DeadlineInference/THIRD_PARTY_LICENSES/manifest.json"}
        if not required.issubset(names):
            raise ValueError("Required release files missing from " + path.name)
        if not source:
            manifest = json.loads(archive.read("DeadlineInference/BUILD_INFO.json"))
            if manifest.get("backend") != "cpu" or manifest.get("weights_included") is not False:
                raise ValueError("Desktop release must identify CPU-only runtime and excluded weights")
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError("ZIP integrity error: " + corrupt)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"archive": path.name, "files": len(names), "bytes": path.stat().st_size, "sha256": digest}


def check_archives(dist):
    manifest = dist/"SHA256SUMS.txt"
    reports = []
    seen = set()
    allowed = {"deadline-inference-source.zip", "DeadlineInference-Windows-x64-CPU.zip"}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name not in allowed or name in seen or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Unexpected or duplicate checksum entry")
        seen.add(name)
        report = check_archive(dist/name, source=name == "deadline-inference-source.zip")
        if report["sha256"] != digest:
            raise ValueError("Checksum mismatch: " + name)
        reports.append(report)
    if not reports:
        raise ValueError("No release archives listed")
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="store_true")
    parser.add_argument("--archives", action="store_true")
    parser.add_argument("--dist", type=Path, default=ROOT/"dist")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.source and not args.archives:
        parser.error("Choose --source, --archives or both")
    report = {"checks_passed": True, "security_audit": False}
    if args.source:
        report["source"] = check_source()
    if args.archives:
        report["archives"] = check_archives(args.dist)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        sys.exit("Release check failed: " + str(exc))
