"""Model registry/checkpoint boundaries; real-model integration is explicit opt-in."""

from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from deadline_engine import backend, models
from deadline_engine.errors import PreparationCancelled


class RegistryTests(unittest.TestCase):
    def test_registry_is_image_only_versioned_and_json_serializable(self):
        records = models.list_models()
        self.assertEqual(len(records), 4)
        self.assertEqual(len({item["id"] for item in records}), 4)
        json.dumps(records)
        for item in records:
            self.assertEqual(item["task"], "image classification")
            self.assertEqual(item["input_channels"], 3)
            self.assertEqual(len(item["sha256"]), 64)
            int(item["sha256"], 16)
            self.assertTrue(item["url"].startswith("https://download.pytorch.org/models/"))
            self.assertTrue(item["filename"].endswith("-" + item["sha256"][:8] + ".pth"))
        records[0]["label"] = "changed copy"
        self.assertEqual(models.get_model_spec()["label"], "MobileNetV3-Small")

    def test_unknown_architecture_and_unbounded_configuration_fail_early(self):
        for args in (
            {"model_id": "arbitrary_python_module"}, {"num_classes": True},
            {"num_classes": 10001}, {"image_size": 32}, {"image_size": 513},
            {"weights_path": None}, {"num_classes": 2},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                models.validate_selection(**args)
        models.validate_selection("resnet18", "my custom weights.pth", 2, 64)

    def test_registry_urls_must_match_installed_torchvision_identity(self):
        for spec in models.list_models():
            weights = SimpleNamespace(url=spec["url"])
            library = SimpleNamespace(models=SimpleNamespace(**{
                spec["weights_enum"]: SimpleNamespace(IMAGENET1K_V1=weights),
            }))
            with self.subTest(model=spec["id"]), patch.object(backend, "_libraries", return_value=(None, library)):
                self.assertEqual(backend._weight_spec(spec["id"])[-1], spec["sha256"])
                weights.url = "https://unrecognized.invalid/" + spec["filename"]
                with self.assertRaisesRegex(RuntimeError, "unsupported.*weight URL"):
                    backend._weight_spec(spec["id"])


class FakeTensor:
    shape = (2, 3)
    layout = "dense"
    is_quantized = False
    dtype = "float32"

    def numel(self):
        return 6

    def element_size(self):
        return 4

    def is_floating_point(self):
        return True


class FakeParameter(FakeTensor):
    def detach(self):
        return FakeTensor()


class CheckpointBoundaryTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".cache"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="test-model-import-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "local weights.pth"
        self.path.write_bytes(b"test state_dict content")
        self.torch = SimpleNamespace(
            Tensor=FakeTensor, strided="dense", load=Mock(return_value={"layer.weight": FakeTensor()}),
            nn=SimpleNamespace(Parameter=FakeParameter),
            isfinite=Mock(return_value=SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True))),
            random=SimpleNamespace(fork_rng=lambda **kwargs: nullcontext()),
            set_num_threads=Mock(),
            manual_seed=Mock(),
        )

    def test_load_is_always_cpu_and_weights_only(self):
        state = backend._read_state_dict(self.path, self.torch)
        self.assertEqual(set(state), {"layer.weight"})
        self.torch.load.assert_called_once_with(self.path.resolve(), map_location="cpu", weights_only=True)

    def test_exact_builtin_parameter_is_detached_but_arbitrary_subclass_is_rejected(self):
        self.torch.load.return_value = {"layer.weight": FakeParameter()}
        state = backend._read_state_dict(self.path, self.torch)
        self.assertIs(type(state["layer.weight"]), FakeTensor)

        class ArbitraryTensorSubclass(FakeTensor):
            pass

        self.torch.load.return_value = {"layer.weight": ArbitraryTensorSubclass()}
        with self.assertRaisesRegex(ValueError, "unsupported field"):
            backend._read_state_dict(self.path, self.torch)

    def test_executable_and_corrupt_checkpoint_never_get_unrestricted_retry(self):
        self.torch.load.side_effect = RuntimeError("not a tensor-only checkpoint")
        with self.assertRaisesRegex(ValueError, "unrestricted pickle are never loaded"):
            backend._read_state_dict(self.path, self.torch)
        self.assertEqual(self.torch.load.call_count, 1)
        self.assertTrue(self.torch.load.call_args.kwargs["weights_only"])

    def test_torchscript_archive_rejected_before_torch_load(self):
        with zipfile.ZipFile(self.path, "w") as handle:
            handle.writestr("model/constants.pkl", b"scripted model")
        with self.assertRaisesRegex(ValueError, "TorchScript/executable"):
            backend._read_state_dict(self.path, self.torch)
        self.torch.load.assert_not_called()

    def test_non_tensor_metadata_and_nan_are_rejected(self):
        for state in ({"epoch": 2}, {"state_dict": {}, "epoch": 2}, {}):
            with self.subTest(state=state):
                self.torch.load.return_value = state
                with self.assertRaises(ValueError):
                    backend._read_state_dict(self.path, self.torch)
        self.torch.load.return_value = {"layer.weight": FakeTensor()}
        self.torch.isfinite.return_value = SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: False))
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            backend._read_state_dict(self.path, self.torch)

    def test_single_state_dict_wrapper_supported(self):
        self.torch.load.return_value = {"state_dict": {"layer.weight": FakeTensor()}}
        self.assertEqual(set(backend._read_state_dict(self.path, self.torch)), {"layer.weight"})

    def test_wrong_architecture_cannot_substitute_random_parameters(self):
        model = SimpleNamespace(state_dict=lambda: {"other.weight": FakeTensor()}, load_state_dict=Mock())
        vision = SimpleNamespace(models=SimpleNamespace(resnet18=lambda **kwargs: model))
        with self.assertRaisesRegex(ValueError, "No random parameters were substituted"):
            backend._load_selected_model(self.torch, vision, self.path, "resnet18", 1000)
        model.load_state_dict.assert_not_called()

    def test_complex_and_float64_dtypes_cannot_be_silently_cast(self):
        for dtype in ("complex64", "float64"):
            tensor = FakeTensor()
            tensor.dtype = dtype
            self.torch.load.return_value = {"layer.weight": tensor}
            model = SimpleNamespace(state_dict=lambda: {"layer.weight": FakeTensor()}, load_state_dict=Mock())
            vision = SimpleNamespace(models=SimpleNamespace(resnet18=lambda **kwargs: model))
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "dtype mismatch"):
                backend._load_selected_model(self.torch, vision, self.path, "resnet18", 1000)
            model.load_state_dict.assert_not_called()

    def test_nonfinite_loaded_model_is_rejected(self):
        state = {"layer.weight": FakeTensor()}
        model = SimpleNamespace(state_dict=lambda: state, load_state_dict=Mock())
        vision = SimpleNamespace(models=SimpleNamespace(resnet18=lambda **kwargs: model))
        self.torch.isfinite.return_value = SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: False))
        with patch.object(backend, "_read_state_dict", return_value=state):
            with self.assertRaisesRegex(ValueError, "Loaded model tensor.*NaN or infinite"):
                backend._load_selected_model(self.torch, vision, self.path, "resnet18", 1000)
        model.load_state_dict.assert_called_once_with(state, strict=True)

    def test_checkpoint_changed_during_load_is_rejected_before_device_move(self):
        engine = backend.TorchBackend("cpu", 1, 7, "resnet18", str(self.path))
        model = SimpleNamespace(eval=Mock())
        with patch.object(backend, "_libraries", return_value=(self.torch, None)), patch.object(
                backend, "_sha256", side_effect=["before", "changed"]), patch.object(
                backend, "_load_selected_model", return_value=model):
            with self.assertRaisesRegex(ValueError, "Checkpoint changed while"):
                engine.prepare()
        model.eval.assert_not_called()
        self.assertFalse(engine.metadata()["prepared"])
        self.assertTrue(engine._lock.acquire(blocking=False))
        engine._lock.release()

    def test_import_copies_verified_bytes_and_records_hash_identity(self):
        with patch.object(backend, "get_cache_root", return_value=self.root), patch.object(
                backend, "_libraries", return_value=(self.torch, None)), patch.object(
                backend, "_load_selected_model", return_value=object()) as validator:
            result = backend.import_weights(self.path, "resnet18", 2, 96)
        destination = Path(result["path"])
        self.assertNotEqual(destination, self.path)
        self.assertEqual(destination.read_bytes(), self.path.read_bytes())
        self.assertEqual(result["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertIn(result["sha256"], destination.name)
        self.assertEqual(result["weights_kind"], "imported_state_dict")
        self.assertEqual(json.loads(destination.with_suffix(".json").read_text())["num_classes"], 2)
        self.assertEqual(validator.call_args.args[-2:], ("resnet18", 2))

    def test_failed_import_cleans_temporary_copy_and_preserves_source(self):
        original = self.path.read_bytes()
        with patch.object(backend, "get_cache_root", return_value=self.root), patch.object(
                backend, "_libraries", return_value=(self.torch, None)), patch.object(
                backend, "_load_selected_model", side_effect=ValueError("architecture mismatch")):
            with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                backend.import_weights(self.path, "resnet18")
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list((self.root / "imported-models").iterdir()), [])


class PreparationCancellationTests(unittest.TestCase):
    def test_cancel_before_preparation_never_imports_or_infers(self):
        engine = backend.TorchBackend("cpu", 2)
        event = threading.Event()
        event.set()
        engine.set_cancel_event(event)
        with patch.object(backend, "_libraries") as libraries, patch.object(engine, "_infer") as inference:
            with self.assertRaises(PreparationCancelled):
                engine.prepare()
        libraries.assert_not_called()
        inference.assert_not_called()
        self.assertFalse(engine.metadata()["prepared"])
        self.assertTrue(engine._lock.acquire(blocking=False))
        engine._lock.release()

    def test_cancel_between_replay_warmups_stops_remaining_forwards(self):
        engine = backend.TorchBackend("cpu", 4)
        event = threading.Event()
        engine.set_cancel_event(event)
        engine._prepared = True  # Existing prepared model; no libraries needed.
        with patch.object(engine, "_infer", side_effect=lambda requests: event.set()) as inference:
            with self.assertRaises(PreparationCancelled):
                engine.prepare()
        self.assertEqual(inference.call_count, 1)
        self.assertEqual(engine.metadata()["prepare_count"], 0)
        self.assertTrue(engine._lock.acquire(blocking=False))
        engine._lock.release()


@unittest.skipUnless(os.environ.get("RUN_MODEL_TESTS") == "1", "set RUN_MODEL_TESTS=1 for real checkpoint import checks")
class RealCheckpointTests(unittest.TestCase):
    def cached(self, model_id):
        spec = models.get_model(model_id)
        path = backend.CHECKPOINTS / spec["filename"]
        self.assertTrue(path.is_file(), f"Explicitly fetch {model_id} before opting into real-model tests.")
        return {"path": str(path), "sha256": spec["sha256"]}

    def test_pretrained_resnet_checkpoint_can_be_imported_and_run(self):
        info = self.cached("resnet18")
        imported = backend.import_weights(info["path"], "resnet18")
        engine = backend.TorchBackend("cpu", 2, 7, "resnet18", imported["path"])
        engine.prepare()
        result = engine.infer([SimpleNamespace(input_index=0), SimpleNamespace(input_index=1)])
        self.assertEqual(len(result["predictions"]), 2)
        metadata = engine.metadata()
        self.assertEqual(metadata["weights_kind"], "imported_state_dict")
        self.assertEqual(metadata["weights_sha256"], info["sha256"])
        self.assertEqual(metadata["parameter_count"], 11689512)
        self.assertGreater(metadata["calibration_s"], 0)

    def test_different_class_count_and_corrupt_files_fail_explicitly(self):
        info = self.cached("squeezenet1_1")
        with self.assertRaisesRegex(ValueError, "does not match"):
            backend.import_weights(info["path"], "resnet18")
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            backend.import_weights(info["path"], "squeezenet1_1", num_classes=3)

    def test_custom_classification_head_imports_with_accurate_provenance(self):
        torch, vision = backend._libraries()
        scratch = Path(__file__).resolve().parents[1] / ".cache"
        with tempfile.TemporaryDirectory(prefix="test-custom-head-", dir=scratch) as temporary:
            path = Path(temporary) / "three-class-test-fixture.pth"
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(11)
                fixture = vision.models.squeezenet1_1(weights=None, num_classes=3)
            # Random weights are only a correctness fixture; metadata must never
            # describe this imported file as pretrained or accuracy validated.
            torch.save(fixture.state_dict(), path)
            imported = backend.import_weights(path, "squeezenet1_1", 3, 64)
            engine = backend.TorchBackend("cpu", 1, 7, "squeezenet1_1", imported["path"], 3, 64)
            engine.prepare()
            prediction = engine.infer([SimpleNamespace(input_index=0, shape=(3, 64, 64))])["predictions"][0]
            self.assertIn(prediction, range(3))
            self.assertEqual(engine.metadata()["weights_kind"], "imported_state_dict")
            self.assertEqual(engine.metadata()["input_shape"], [3, 64, 64])
            self.assertIn("training and accuracy not verified", engine.metadata()["training_provenance"])


if __name__ == "__main__":
    unittest.main()
