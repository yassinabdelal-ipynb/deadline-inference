"""Real pretrained inference, explicit weight acquisition, and runtime calibration.

Synthetic images exercise scheduling; their predicted classes say nothing about
model accuracy. Construct and prepare one backend, then give it to one inference
worker. The lock catches accidental overlapping model use.
"""

from __future__ import annotations

import hashlib
import os
import re
import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
from urllib.request import urlopen
import zipfile

from .paths import get_cache_root
from .models import DEFAULT_MODEL, MAX_IMPORTED_WEIGHT_BYTES, get_model, validate_selection
from .errors import PreparationCancelled

TORCH_CACHE = get_cache_root() / "torch"
CHECKPOINTS = TORCH_CACHE / "hub" / "checkpoints"
MAX_WEIGHT_BYTES = 32 * 1024 * 1024
EXPECTED_WEIGHT_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"


def _libraries():
    try:
        import torch
        import torchvision
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "PyTorch/torchvision could not be imported. Use the project setup "
            "instructions for your operating system, then run the system check. "
            "Original error: " + str(exc)
        ) from exc
    return torch, torchvision


def _weight_spec(model_id=DEFAULT_MODEL):
    torch, torchvision = _libraries()
    spec = get_model(model_id)
    weights = getattr(torchvision.models, spec["weights_enum"]).IMAGENET1K_V1
    filename = Path(urlparse(weights.url).path).name
    match = re.search(r"-([a-f0-9]+)\.pth$", filename)
    if (match is None or filename != spec["filename"] or weights.url != spec["url"]
            or not spec["sha256"].startswith(match.group(1))):
        raise RuntimeError(
            f"The installed torchvision has an unsupported {spec['label']} weight URL. "
            "Use this project's pinned dependencies; review and pin the full weight "
            "SHA256 before using a different checkpoint."
        )
    return torch, torchvision, weights, CHECKPOINTS / filename, spec["sha256"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_weights(model_id=DEFAULT_MODEL) -> dict:
    """Explicitly download the named torchvision weights, once, with hash checking.

    The URL comes from the pinned torchvision installation; its expected full
    SHA256 is pinned here and checked against that versioned filename.
    Merely constructing/preparing a backend never downloads anything. Network
    operations use a 30-second socket timeout and an architecture-specific size
    cap; a verified temporary file is moved into the cache.
    """
    spec = get_model(model_id)
    _, torchvision, weights, path, prefix = _weight_spec(model_id)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = _sha256(path)
        if not actual.startswith(prefix):
            raise RuntimeError(
                f"Cached weights failed SHA256 verification: {path}. "
                "Remove this damaged file, then explicitly fetch weights again."
            )
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=path.name + ".", suffix=".download", dir=CHECKPOINTS, delete=False
            ) as handle:
                temporary = Path(handle.name)
            with urlopen(weights.url, timeout=30) as response, temporary.open("wb") as output:
                downloaded = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    downloaded += len(chunk)
                    limit = MAX_WEIGHT_BYTES if model_id == DEFAULT_MODEL else spec["max_weight_bytes"]
                    if downloaded > limit:
                        raise RuntimeError(f"Weight download exceeds this model's {limit // 1024**2} MiB size limit.")
                    output.write(chunk)
            actual = _sha256(temporary)
            if not actual.startswith(prefix):
                raise RuntimeError("Downloaded weights failed SHA256 verification.")
            os.replace(temporary, path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not fetch pretrained weights from {weights.url}. "
                "Check network access and retry the explicit fetch-weights command. "
                f"No randomly initialized model was substituted. Details: {exc}"
            ) from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {
        "model": model_id,
        "weights": spec["weights_name"],
        "url": weights.url,
        "path": str(path),
        "sha256": actual,
        "expected_sha256": prefix,
        "torchvision_version": torchvision.__version__,
    }


def _read_state_dict(path, torch):
    """Read plain tensor parameters, never executable modules or model code.

    weights_only reduces pickle risk; it is not a sandbox. Only import files
    whose source you trust. File and tensor limits constrain ordinary mistakes.
    """
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("Choose a .pt or .pth tensor state_dict checkpoint, not a Python model or executable.")
    if not path.is_file():
        raise ValueError(f"Checkpoint does not exist: {path}")
    if not 0 < path.stat().st_size <= MAX_IMPORTED_WEIGHT_BYTES:
        raise ValueError("Checkpoint must be nonempty and at most 256 MiB.")
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                if len(entries) > 10000 or sum(item.file_size for item in entries) > MAX_IMPORTED_WEIGHT_BYTES:
                    raise ValueError("Checkpoint archive exceeds the tensor checkpoint size limit.")
                if any("/code/" in item.filename or item.filename.endswith("/constants.pkl") for item in entries):
                    raise ValueError("TorchScript/executable models are unsupported. Export model.state_dict() instead.")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and len(state) == 1:
            for wrapper in ("state_dict", "model_state_dict"):
                if wrapper in state:
                    state = state[wrapper]
                    break
        if not isinstance(state, dict) or not state:
            raise ValueError("Checkpoint must be a nonempty dictionary of named tensors.")
        total_bytes = 0
        # Some official older torchvision checkpoints (including ResNet18 V1)
        # store built-in nn.Parameter values. They contain tensor data, not
        # model code; accept that exact built-in type, never arbitrary subclasses.
        parameter_type = getattr(getattr(torch, "nn", None), "Parameter", None)
        for name, tensor in state.items():
            if (not isinstance(name, str) or type(tensor) not in (torch.Tensor, parameter_type)
                    or tensor.layout != torch.strided or tensor.is_quantized):
                raise ValueError(f"Only string keys and plain dense, nonquantized tensors/Parameters are supported; unsupported field {name!r} ({type(tensor).__name__}).")
            total_bytes += tensor.numel() * tensor.element_size()
            if total_bytes > MAX_IMPORTED_WEIGHT_BYTES:
                raise ValueError("Checkpoint tensor data exceeds 256 MiB.")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
                raise ValueError(f"Checkpoint tensor {name!r} contains NaN or infinite values.")
            if parameter_type is not None and type(tensor) is parameter_type:
                state[name] = tensor.detach()
        return state
    except (ValueError, OSError) as exc:
        raise ValueError(f"Cannot read image model checkpoint {path.name}: {exc}") from exc
    except Exception as exc:
        raise ValueError(
            f"Cannot read image model checkpoint {path.name}. Use torch.save(model.state_dict(), path) "
            "for a supported architecture. Executable modules and unrestricted pickle are never loaded. "
            f"Details: {exc}"
        ) from exc


def _load_selected_model(torch, torchvision, path, model_id, num_classes):
    state = _read_state_dict(path, torch)
    # Model constructors initialize tensors before loading. Restoring the RNG
    # prevents validation/import from changing another experiment's input seed.
    with torch.random.fork_rng(devices=[]):
        model = getattr(torchvision.models, model_id)(weights=None, num_classes=num_classes)
    expected = model.state_dict()
    all_missing = sorted(set(expected) - set(state))
    # Old torchvision BatchNorm checkpoints legitimately predate this integer
    # training counter. PyTorch's strict loader supplies a zero counter only
    # when the checkpoint's module version permits that compatibility rule.
    # It is unused by eval-mode forward. Learned parameters are never filled.
    missing = [name for name in all_missing if not name.endswith(".num_batches_tracked")]
    legacy_counters = [name for name in all_missing if name.endswith(".num_batches_tracked")]
    extra = sorted(set(state) - set(expected))
    mismatched = [name for name in expected.keys() & state.keys()
                  if tuple(expected[name].shape) != tuple(state[name].shape)]
    mismatched_dtypes = [name for name in expected.keys() & state.keys()
                         if expected[name].dtype != state[name].dtype]
    if missing or extra or mismatched or mismatched_dtypes:
        detail = []
        if missing:
            detail.append("missing keys: " + ", ".join(missing[:3]))
        if extra:
            detail.append("unexpected keys: " + ", ".join(extra[:3]))
        if mismatched:
            detail.append("shape mismatch: " + ", ".join(mismatched[:3]))
        if mismatched_dtypes:
            detail.append("dtype mismatch: " + ", ".join(
                f"{name} ({state[name].dtype}, expected {expected[name].dtype})"
                for name in mismatched_dtypes[:3]))
        raise ValueError(
            f"Checkpoint does not match {get_model(model_id)['label']} with {num_classes} classes "
            f"({'; '.join(detail)}). Select the architecture and class count used to train it. "
            "No random parameters were substituted."
        )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint does not strictly match {get_model(model_id)['label']} with {num_classes} classes: {exc}. No random parameters were substituted.") from exc
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"Loaded model tensor {name!r} contains NaN or infinite values.")
    model._deadline_checkpoint_info = {"legacy_batchnorm_counters_initialized": legacy_counters}
    return model


def import_weights(source, model_id=DEFAULT_MODEL, num_classes=1000, image_size=224) -> dict:
    """Validate a trusted local state_dict and copy it into managed model storage.

    No upload or network service is involved. Source bytes are copied first,
    then validated and identified; edits to the original file cannot alter the
    imported experiment. A sidecar records identity and import limitations.
    """
    validate_selection(model_id, str(source), num_classes, image_size)
    source = Path(source).expanduser().resolve()
    if source.suffix.lower() not in {".pth", ".pt"} or not source.is_file():
        raise ValueError("Choose an existing .pt or .pth tensor state_dict checkpoint.")
    if not 0 < source.stat().st_size <= MAX_IMPORTED_WEIGHT_BYTES:
        raise ValueError("Checkpoint must be nonempty and at most 256 MiB.")
    destination = get_cache_root() / "imported-models"
    destination.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="checkpoint-", suffix=source.suffix.lower(), dir=destination, delete=False) as output:
            temporary = Path(output.name)
            with source.open("rb") as handle:
                written = 0
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    written += len(chunk)
                    if written > MAX_IMPORTED_WEIGHT_BYTES:
                        raise ValueError("Checkpoint grew beyond the 256 MiB import size limit.")
                    output.write(chunk)
        torch, torchvision = _libraries()
        torch.set_num_threads(2)
        model = _load_selected_model(torch, torchvision, temporary, model_id, num_classes)
        del model
        digest = _sha256(temporary)
        target = destination / f"{model_id}-{digest}.pth"
        os.replace(temporary, target)
        result = {
            "path": str(target), "sha256": digest, "model": model_id,
            "num_classes": num_classes, "image_size": image_size,
            "source_name": source.name, "weights_kind": "imported_state_dict",
            "training_provenance": "user supplied; training and accuracy not verified",
        }
        target.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def process_peak_memory_bytes() -> int | None:
    """OS process peak resident memory (not Python allocations alone)."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if psapi.GetProcessMemoryInfo(
                kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            ):
                return int(counters.PeakWorkingSetSize)
        except (OSError, AttributeError):
            return None
    else:
        try:
            import resource
            import sys

            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(value if sys.platform == "darwin" else value * 1024)
        except (ImportError, OSError):
            pass
    return None


class TorchBackend:
    """One supported eval-mode image classifier on one explicit device.

    Runtime estimates use the slowest of three calibrated end-to-end host
    service observations per batch size, multiplied by 1.25, plus 1 ms. They
    are scheduling heuristics, never guarantees. CPU runs use two torch threads.
    """

    def __init__(self, device: str = "cuda", max_batch: int = 8, seed: int = 7,
                 model_id: str = DEFAULT_MODEL, weights_path: str = "",
                 num_classes: int = 1000, image_size: int = 224):
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if isinstance(max_batch, bool) or not isinstance(max_batch, int) or not 1 <= max_batch <= 64:
            raise ValueError("max_batch must be an integer from 1 through 64")
        self.device = device
        validate_selection(model_id, weights_path, num_classes, image_size)
        self.model_id = model_id
        self.weights_path = weights_path
        self.num_classes = num_classes
        self.image_size = image_size
        self.input_shape = (3, image_size, image_size)
        self.max_batch = max_batch
        self.seed = seed
        self.estimates_s: dict[int, float] = {}
        self.calibration: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._prepared = False
        self._model = None
        self._pool = None
        self._torch = None
        self._info: dict = {}
        self._prepare_count = 0
        self._cancel_event = None

    def set_cancel_event(self, event) -> None:
        """Attach one replay's cooperative stop signal before preparation.

        Running native kernels/library imports finish before the next check.
        This does not interrupt a forward or mutate a model concurrently.
        """
        if event is not None and not callable(getattr(event, "is_set", None)):
            raise ValueError("Cancellation event must provide is_set().")
        self._cancel_event = event

    def _check_cancel(self) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise PreparationCancelled("Model preparation cancelled between operations.")

    def prepare(self) -> None:
        """Load only cached weights, build inputs, warm up and calibrate."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("The inference backend is already in use.")
        try:
            preparation_start = time.perf_counter()
            self._check_cancel()
            if self._prepared:
                # Each policy/repeat gets identical fresh warmup on its own
                # inference worker. Preserve the common scheduling calibration.
                for batch_size in range(1, self.max_batch + 1):
                    requests = [SimpleNamespace(input_index=i) for i in range(batch_size)]
                    for _ in range(2):
                        self._check_cancel()
                        self._infer(requests)
                self._check_cancel()
                self.reset_peak_memory()
                self._prepare_count += 1
                self._info["last_replay_warmup_s"] = time.perf_counter() - preparation_start
                return
            spec = get_model(self.model_id)
            if self.weights_path:
                torch, torchvision = _libraries()
                path = Path(self.weights_path).expanduser().resolve()
                prefix = None
                weights_url = None
                weights_name = "User-supplied tensor state_dict"
                weights_kind = "imported_state_dict"
            else:
                torch, torchvision, weights, path, prefix = _weight_spec(self.model_id)
                weights_url = weights.url
                weights_name = spec["weights_name"]
                weights_kind = "official_pretrained"
            self._check_cancel()
            if not path.is_file():
                raise RuntimeError(
                    f"Pretrained weights are missing: {path}. Run the explicit "
                    "Fetch model action in the desktop app, or run "
                    f"python -m deadline_engine fetch-weights --model {self.model_id} in the project environment. "
                    "Inference never downloads weights or substitutes a random model."
                )
            actual = _sha256(path)
            if prefix is not None and not actual.startswith(prefix):
                raise RuntimeError(
                    f"Cached weights failed SHA256 verification: {path}. Remove "
                    "the damaged file and explicitly fetch weights again."
                )
            if self.device == "cuda" and not torch.cuda.is_available():
                if torch.version.cuda is None:
                    route = ("This standalone app contains CPU-only PyTorch. Use Run Deadline Inference.cmd "
                             "in a source project with CUDA dependencies installed."
                             if getattr(sys, "frozen", False) else
                             "This project environment contains CPU-only PyTorch. Run "
                             "scripts/setup.ps1 -Backend cuda, then reopen Run Deadline Inference.cmd.")
                    raise RuntimeError("CUDA cannot run in this CPU-only runtime. " + route + " CPU testing remains available.")
                raise RuntimeError(
                    "CUDA-enabled PyTorch is installed, but no usable NVIDIA CUDA device was detected. "
                    "Open System check for hardware and driver details. CPU testing remains available."
                )
            torch.set_num_threads(2)
            torch.manual_seed(self.seed)
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            self._torch = torch
            self._check_cancel()
            self._model = _load_selected_model(torch, torchvision, path, self.model_id, self.num_classes)
            if _sha256(path) != actual:
                raise ValueError("Checkpoint changed while it was being loaded. Stop any writer and import a stable checkpoint before retrying.")
            self._check_cancel()
            self._model.eval().to(self.device)
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            self.pool_size = max(32, self.max_batch * 2)
            pool = torch.rand((self.pool_size, *self.input_shape), generator=generator)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self._pool = ((pool - mean) / std).contiguous()
            if self.device == "cuda":
                self._pool = self._pool.pin_memory()
            self._info = {
                "backend": "pytorch",
                "device": self.device,
                "device_name": torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU",
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "model": self.model_id,
                "model_label": spec["label"],
                "task": "image classification",
                "num_classes": self.num_classes,
                "weights": weights_name,
                "weights_kind": weights_kind,
                "weights_url": weights_url,
                "weights_sha256": actual,
                "weights_expected_sha256": prefix,
                "weights_path": str(path),
                "input_shape": list(self.input_shape),
                "input_dtype": "float32",
                "input_pool_size": self.pool_size,
                "input_seed": self.seed,
                "input_description": "Procedural random RGB images, ImageNet normalized; predictions do not evaluate accuracy",
                "torch_cpu_threads": torch.get_num_threads(),
                "max_batch": self.max_batch,
                "parameter_count": sum(parameter.numel() for parameter in self._model.parameters()),
                "model_parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in self._model.parameters()),
                "model_buffer_bytes": sum(buffer.numel() * buffer.element_size() for buffer in self._model.buffers()),
                "checkpoint_bytes": path.stat().st_size,
                "training_provenance": "official torchvision ImageNet-1K pretrained weights" if not self.weights_path else "user supplied; training and accuracy not verified",
                "checkpoint_compatibility": getattr(self._model, "_deadline_checkpoint_info", {}),
                "estimate_rule": "max of 3 host service samples * 1.25 + 0.001 seconds after 2 warmups per batch size",
                "device_timing": "CUDA events around model forward only" if self.device == "cuda" else "host perf_counter around model forward only",
                "service_timing": "host perf_counter including input assembly, transfer, forward, prediction transfer and synchronization",
                "cuda_deterministic_cudnn": self.device == "cuda",
            }
            if self.device == "cuda":
                props = torch.cuda.get_device_properties(0)
                self._info.update({
                    "total_vram_bytes": props.total_memory,
                    "compute_capability": list(torch.cuda.get_device_capability(0)),
                })
            calibration_start = time.perf_counter()
            self._info["model_setup_s"] = calibration_start - preparation_start
            for batch_size in range(1, self.max_batch + 1):
                requests = [SimpleNamespace(input_index=i) for i in range(batch_size)]
                for _ in range(2):
                    self._check_cancel()
                    self._infer(requests)
                samples = []
                for _ in range(3):
                    self._check_cancel()
                    samples.append(self._infer(requests))
                service = [sample["service_ms"] for sample in samples]
                estimate = max(service) / 1000 * 1.25 + 0.001
                self.estimates_s[batch_size] = estimate
                self.calibration[batch_size] = {
                    "host_service_ms": service,
                    "device_forward_ms": [sample["device_ms"] for sample in samples],
                    "estimate_s": estimate,
                    "sample_count": 3,
                    "warmup_count": 2,
                }
            self._check_cancel()
            if self.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            self._prepared = True
            self._prepare_count += 1
            self._info["calibration_s"] = time.perf_counter() - calibration_start
            self._info["initial_preparation_s"] = time.perf_counter() - preparation_start
        finally:
            self._lock.release()

    def estimate_s(self, batch_size: int) -> float:
        if not self._prepared:
            raise RuntimeError("Call backend.prepare() before requesting a runtime estimate.")
        if batch_size not in self.estimates_s:
            raise ValueError(f"batch size must be from 1 through {self.max_batch}")
        return self.estimates_s[batch_size]

    def _infer(self, requests: list) -> dict:
        torch = self._torch
        start = time.perf_counter()
        for request in requests:
            shape = getattr(request, "shape", self.input_shape)
            dtype = getattr(request, "dtype", "float32")
            if not isinstance(shape, (tuple, list)) or tuple(shape) != self.input_shape or dtype != "float32":
                raise ValueError(
                    f"This image backend accepts only shape {self.input_shape} "
                    "with dtype float32; request metadata must match its actual inputs."
                )
        indices = [request.input_index for request in requests]
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
            raise ValueError("input_index must be a nonnegative integer")
        # The modulo mapping is stable across both policies and arbitrary trace sizes.
        batch = torch.stack([self._pool[index % self.pool_size] for index in indices])
        with torch.inference_mode():
            batch = batch.to(self.device, non_blocking=False)
            if self.device == "cuda":
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                event_start.record()
                logits = self._model(batch)
                event_end.record()
                event_end.synchronize()
                device_ms = event_start.elapsed_time(event_end)
            else:
                forward_start = time.perf_counter()
                logits = self._model(batch)
                device_ms = (time.perf_counter() - forward_start) * 1000
            if logits.ndim != 2 or tuple(logits.shape) != (len(requests), self.num_classes):
                raise RuntimeError("Image classifier must return one class-score vector per input.")
            predictions = logits.argmax(dim=1).cpu().tolist()
        return {
            "device_ms": float(device_ms),
            "service_ms": (time.perf_counter() - start) * 1000,
            "predictions": predictions,
        }

    def infer(self, requests: list) -> dict:
        if not self._prepared:
            raise RuntimeError("Call backend.prepare() before inference.")
        if not 1 <= len(requests) <= self.max_batch:
            raise ValueError(f"A batch must contain 1 through {self.max_batch} requests")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent inference on the same backend is unsupported.")
        try:
            return self._infer(requests)
        finally:
            self._lock.release()

    def metadata(self) -> dict:
        return {
            **self._info,
            "prepared": self._prepared,
            "prepare_count": self._prepare_count,
            "warmup_per_replay": "two unmeasured forwards per batch size; shared initial calibration retained",
            "calibration": {str(size): dict(sample) for size, sample in self.calibration.items()},
        }

    def reset_peak_memory(self) -> None:
        """Reset CUDA allocation high-water marks after that replay's warmup.

        The caching allocator may retain warmup blocks. Reserved memory includes
        those persistent blocks, and CPU process peak cannot be reset portably.
        """
        if self._torch is not None and self.device == "cuda":
            self._torch.cuda.synchronize()
            self._torch.cuda.reset_peak_memory_stats()

    def peak_memory(self) -> dict:
        return {
            "cuda_peak_allocated_bytes": int(self._torch.cuda.max_memory_allocated())
            if self._prepared and self.device == "cuda" else None,
            "cuda_peak_reserved_bytes": int(self._torch.cuda.max_memory_reserved())
            if self._prepared and self.device == "cuda" else None,
            "process_peak_resident_bytes": process_peak_memory_bytes(),
            "process_peak_scope": "process lifetime high-water mark; includes preparation and earlier runs",
            "cuda_peak_scope": "since replay warmup; reserved bytes include cached warmup blocks",
        }
