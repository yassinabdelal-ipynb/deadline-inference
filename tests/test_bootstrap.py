"""Setup must preserve an existing CUDA environment unless replacement is explicit."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("bootstrap", Path(__file__).resolve().parents[1]/"scripts"/"bootstrap.py")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class BackendPreservationTests(unittest.TestCase):
    def test_new_environment_and_same_backend_need_no_reinstallation(self):
        for current in (None, {"backend": "cuda", "torch": "2.12.0+cu130"}):
            with self.subTest(current=current), patch.object(bootstrap, "installed_backend", return_value=current):
                self.assertFalse(bootstrap.backend_change(Path("folder with spaces/python.exe"), "cuda"))

    def test_default_cpu_cannot_replace_existing_cuda(self):
        with patch.object(bootstrap, "installed_backend", return_value={"backend": "cuda", "torch": "2.12.0+cu130"}):
            with self.assertRaisesRegex(ValueError, "Refusing to replace.*--replace-backend"):
                bootstrap.backend_change(Path("python.exe"), "cpu")
            self.assertTrue(bootstrap.backend_change(Path("python.exe"), "cpu", replace=True))

    def test_explicit_cuda_cannot_replace_cpu_without_flag(self):
        with patch.object(bootstrap, "installed_backend", return_value={"backend": "cpu", "torch": "2.12.0+cpu"}):
            with self.assertRaisesRegex(ValueError, "separate installation"):
                bootstrap.backend_change(Path("python.exe"), "cuda")
            self.assertTrue(bootstrap.backend_change(Path("python.exe"), "cuda", replace=True))

    def test_probe_passes_interpreter_path_as_one_argument(self):
        interpreter = Path("folder with spaces")/"python.exe"
        with patch.object(bootstrap.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, '{"backend":"cpu","torch":"test"}')) as run:
            self.assertEqual(bootstrap.installed_backend(interpreter)["backend"], "cpu")
        self.assertEqual(run.call_args.args[0][0], str(interpreter))
        self.assertTrue(run.call_args.kwargs["check"])

    def test_failed_probe_never_assumes_environment_is_empty(self):
        with patch.object(bootstrap.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "probe")):
            with self.assertRaises(subprocess.CalledProcessError):
                bootstrap.backend_change(Path("python.exe"), "cpu")


if __name__ == "__main__":
    unittest.main()
