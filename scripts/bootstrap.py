"""Cross-platform project-local setup. No activation or system changes.

python scripts/bootstrap.py --backend cpu --download-weights
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def installed_backend(python):
    """Inspect an existing environment before changing its execution backend."""
    probe = subprocess.run([str(python), "-c",
        "import importlib.util,json; found=importlib.util.find_spec('torch'); "
        "exec(\"if found:\\n import torch\\n print(json.dumps({'backend':'cuda' if torch.version.cuda else 'cpu','torch':torch.__version__}))\\nelse:\\n print('null')\")"],
        capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(probe.stdout.strip())


def backend_change(python, requested, replace=False):
    current = installed_backend(python)
    changed = current is not None and current["backend"] != requested
    if changed and not replace:
        raise ValueError(f"This environment already contains {current['torch']} ({current['backend']}). "
                         f"Refusing to replace it with {requested}. Use --backend {current['backend']} to keep it, "
                         "--environment with another folder for a separate installation, or --replace-backend "
                         "to deliberately change this environment.")
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--environment", type=Path, default=ROOT / ".venv")
    parser.add_argument("--replace-backend", action="store_true", help="Explicitly replace an existing CPU/CUDA backend instead of preserving it")
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 14) or sys.maxsize <= 2**32:
        parser.error("Use 64-bit Python 3.14. Install it from python.org, then rerun this script.")
    if args.backend == "cuda" and (sys.platform == "darwin" or platform.machine().lower() not in ("amd64", "x86_64")):
        parser.error("This CUDA setup targets Windows/Linux x64 NVIDIA PCs. Choose --backend cpu here.")
    env = args.environment.expanduser().resolve()
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(env)
    try:
        replacing = backend_change(python, args.backend, args.replace_backend)
    except ValueError as exc:
        parser.error(str(exc))
    package_file = ROOT / ("requirements-lock.txt" if args.backend == "cuda" else "requirements-cpu.txt")
    child_env = {**os.environ, "UV_CACHE_DIR": str(ROOT / ".cache" / "uv")}
    index = "https://download.pytorch.org/whl/" + ("cu130" if args.backend == "cuda" else "cpu")
    # macOS CPU wheels use PyPI, whereas the official CPU index serves Windows/Linux.
    index = "https://pypi.org/simple" if sys.platform == "darwin" else index
    installer = ([shutil.which("uv"), "pip", "install", "--python", str(python)] if shutil.which("uv")
                 else [str(python), "-m", "pip", "install"])
    if replacing:
        installer.extend(["--reinstall-package", "torch", "--reinstall-package", "torchvision"]
                         if shutil.which("uv") else ["--force-reinstall"])
    if not shutil.which("uv"):
        subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], check=True, cwd=ROOT, env=child_env)
    subprocess.run([*installer, "-r", str(package_file), "--index-url", index], check=True, cwd=ROOT, env=child_env)
    subprocess.run([str(python), "-c", "import tkinter; print('Native Tk', tkinter.TkVersion)"], check=True, cwd=ROOT, env=child_env)
    if args.download_weights:
        subprocess.run([str(python), "-m", "deadline_engine", "fetch-weights"], check=True, cwd=ROOT, env=child_env)
    subprocess.run([str(python), "-m", "deadline_engine", "doctor", "--output", "results/environment.json"], check=True, cwd=ROOT, env=child_env)
    print(f"Ready. Start the native app:\n  \"{python}\" -m deadline_engine desktop")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Setup stopped (exit {exc.returncode}). Check the error above, network access and Tcl/Tk installation. Rerunning is safe.")
