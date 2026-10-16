"""Build a Windows CPU desktop bundle and a clean source ZIP, locally only."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SINGLES = {
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", "SECURITY.md",
    "CHANGELOG.md", "CODE_OF_CONDUCT.md", "CITATION.cff", "pyproject.toml",
    "requirements-cpu.txt", "requirements-lock.txt", "requirements-build.txt", ".gitignore",
    "STATUS.md", ".gitattributes", "Run Deadline Inference.cmd", "Run GPU Inference.cmd",
}
SOURCE_TREES = {"deadline_engine", "scripts", "tests", "docs", ".github", "licenses"}
SOURCE_SUFFIXES = {".py", ".ps1", ".cmd", ".md", ".txt", ".yml", ".yaml", ".json", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".terms", ".cff"}


def source_allowed(relative):
    relative = Path(relative)
    parts = relative.parts
    if not parts or relative.is_absolute() or ".." in parts:
        return False
    if len(parts) == 1:
        return relative.name in SOURCE_SINGLES
    if parts[0] == "docs" and relative.name in {"RESULTS.md", "REVISION_CPU_RESULTS.md", "ENGINEERING_LOG.md"}:
        return False
    if parts[:2] == ("docs", "public") and relative.name in {"profiles.csv", "replays.csv"}:
        return True
    if parts[0] == "results":
        return (len(parts) == 6 and parts[1] == "image-models"
                and parts[-2:] in (("comparison", "comparison.csv"), ("profile", "profile.csv")))
    return (parts[0] in SOURCE_TREES and relative.suffix.lower() in SOURCE_SUFFIXES
            and not any(part.startswith(".") or part == "__pycache__" for part in parts[1:]))


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_licenses(target):
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name", "unknown")
        copied = []
        for file in distribution.files or []:
            if any(token in Path(str(file)).name.lower() for token in ("license", "licence", "copying", "notice")):
                source = Path(distribution.locate_file(file))
                if source.is_file():
                    relative = Path(name) / str(file).replace("../", "").replace("..\\", "")
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    copied.append(str(relative))
        manifest.append({"name": name, "version": distribution.version,
                         "license_expression": distribution.metadata.get("License-Expression"), "files": copied})
    python_root = Path(sys.base_prefix)
    for source in [python_root / "LICENSE.txt", *python_root.glob("tcl/**/license.terms")]:
        if source.is_file():
            destination = target / "Python-Tcl-Tk" / source.relative_to(python_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for source in (ROOT/"licenses").glob("*"):
        if source.is_file():
            shutil.copy2(source, target/source.name)


def source_files():
    files = [ROOT / name for name in SOURCE_SINGLES if (ROOT/name).is_file()]
    for directory in (*SOURCE_TREES, "results/image-models"):
        files.extend(p for p in (ROOT/directory).rglob("*") if p.is_file() and source_allowed(p.relative_to(ROOT)))
    return sorted(set(files))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true", help="Package an already built app")
    args = parser.parse_args(argv)
    dist = ROOT / "dist"
    if not dist.resolve().is_relative_to(ROOT):
        raise SystemExit("Build output must remain inside this project.")
    dist.mkdir(exist_ok=True)
    # Keep an open previous executable usable while a new release is built.
    version = tomllib.loads((ROOT/"pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if not version or any(c not in "0123456789.-abcdefghijklmnopqrstuvwxyz" for c in version):
        raise SystemExit("Invalid project version for build directory")
    bundle_dist = dist / ("v" + version)
    bundle_dist.mkdir(exist_ok=True)
    temporary = ROOT / ".cache" / "build-temp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["PYINSTALLER_CONFIG_DIR"] = str(ROOT / ".cache" / "pyinstaller")
    os.environ["TEMP"] = str(temporary)
    os.environ["TMP"] = str(temporary)
    if not args.source_only:
        if os.name != "nt":
            parser.error("The executable target is Windows. Build on Windows; use --source-only here.")
        import torch
        if torch.version.cuda is not None:
            parser.error("Use the separate CPU build environment. A CUDA bundle would needlessly restrict distribution and increase size.")
        if not args.skip_build:
            subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--windowed",
                "--noupx", "--name", "DeadlineInference", "--paths", str(ROOT),
                "--distpath", str(bundle_dist), "--workpath", str(ROOT/".build"), "--specpath", str(ROOT/".build"),
                "--collect-all", "torchvision", "--copy-metadata", "torch", "--copy-metadata", "torchvision",
                "--exclude-module", "tensorboard", "--exclude-module", "matplotlib", "--exclude-module", "IPython",
                str(ROOT/"scripts"/"desktop_entry.py")], check=True, cwd=ROOT)
        bundle = bundle_dist / "DeadlineInference"
        if not (bundle / "DeadlineInference.exe").is_file():
            raise SystemExit("Build output is missing. Rerun without --skip-build.")
        collect_licenses(bundle / "THIRD_PARTY_LICENSES")
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "README.md"):
            shutil.copy2(ROOT/name, bundle/name)
        shutil.copytree(ROOT/"docs", bundle/"docs", dirs_exist_ok=True)
        manifest = {"version": version, "platform": "Windows x64", "python": sys.version, "torch": torch.__version__,
                    "backend": "cpu", "weights_included": False, "signing": "unsigned local build",
                    "dependencies": {d.metadata["Name"]: d.version for d in metadata.distributions()}}
        (bundle / "BUILD_INFO.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with zipfile.ZipFile(dist/"DeadlineInference-Windows-x64-CPU.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(bundle_dist))
    with zipfile.ZipFile(dist/"deadline-inference-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
            archive.write(path, Path("deadline-inference") / path.relative_to(ROOT))
    archives = [dist/"deadline-inference-source.zip"]
    if not args.source_only:
        archives.append(dist/"DeadlineInference-Windows-x64-CPU.zip")
    archives.sort()
    (dist/"SHA256SUMS.txt").write_text("".join(f"{checksum(p)}  {p.name}\n" for p in archives), encoding="utf-8")
    print("\n".join(f"{p.name}: {p.stat().st_size/1048576:.1f} MiB" for p in archives))


if __name__ == "__main__":
    main()
