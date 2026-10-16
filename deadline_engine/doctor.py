"""Read-only environment diagnostics; never install or change system settings."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from .paths import get_cache_root, get_data_root


def _command(command: list[str], timeout: float = 15) -> dict:
    executable = shutil.which(command[0])
    if executable is None:
        return {"command": command, "available": False}
    try:
        completed = subprocess.run(
            [executable, *command[1:]], capture_output=True, text=True,
            errors="replace", timeout=timeout, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return {
            "command": command, "available": True, "path": executable,
            "exit_code": completed.returncode,
            "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "available": True, "path": executable, "error": str(exc)}


def diagnose() -> dict:
    """Return JSON-compatible actual environment observations and clear caveats."""
    info = {
        "os": {
            "system": platform.system(), "release": platform.release(),
            "version": platform.version(), "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "python": {
            "version": sys.version, "executable": sys.executable,
            "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "in_virtualenv": sys.prefix != sys.base_prefix,
        },
        "cpu": {"processor": platform.processor(), "logical_processors": os.cpu_count()},
        "workspace": str(Path(__file__).resolve().parents[1]),
        "frozen_app": bool(getattr(sys, "frozen", False)),
        "cache_root": str(get_cache_root()),
        "data_root": str(get_data_root()),
        "runtime_scope": (
            "Diagnostics describe the Windows host executing this process. "
            "A local path alone does not prove this is a particular laptop; "
            "compare the reported GPU and OS with your computer."
        ),
        "commands": {
            name: _command(args) for name, args in {
                "python_launcher": ["py", "--list-paths"],
                **({} if getattr(sys, "frozen", False) else {"pip": [sys.executable, "-m", "pip", "--version"]}),
                "git": ["git", "--version"],
                "powershell": ["powershell", "-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"],
                "nvcc": ["nvcc", "--version"],
                "cl": ["cl"],
                "gcc": ["gcc", "--version"],
                "clang": ["clang", "--version"],
                "node": ["node", "--version"],
                "nvidia_smi": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"],
                "nvidia_smi_full": ["nvidia-smi"],
            }.items()
        },
        "torch": {},
        "caveats": [
            "nvidia-smi's CUDA Version is the driver's supported CUDA level, not proof of an installed toolkit or a working PyTorch CUDA runtime.",
            "Missing nvcc/system C++ compilers does not prevent using compatible prebuilt PyTorch wheels; this project does not compile custom CUDA kernels.",
            "System check is read-only: it does not install drivers, compilers, WSL, change execution policy, or reboot.",
            "Peak VRAM and successful model execution must be measured by a real inference run, separately from this diagnostic.",
        ],
    }
    try:
        import torch
        info["torch"] = {
            "version": torch.__version__, "path": torch.__file__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cudnn_version": torch.backends.cudnn.version(),
            "supported_cuda_architectures": torch.cuda.get_arch_list(),
            "devices": [],
        }
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                info["torch"]["devices"].append({
                    "index": index, "name": props.name,
                    "total_vram_bytes": props.total_memory,
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                })
    except Exception as exc:
        info["torch"] = {"import_error": repr(exc)}
    try:
        import torchvision
        info["torchvision"] = {"version": torchvision.__version__, "path": torchvision.__file__}
    except Exception as exc:
        info["torchvision"] = {"import_error": repr(exc)}
    info["weights_cached"] = [str(path) for path in (get_cache_root() / "torch" / "hub" / "checkpoints").glob("*.pth")]
    return info


def main() -> None:
    print(json.dumps(diagnose(), indent=2))


if __name__ == "__main__":
    main()
