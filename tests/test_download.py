"""Explicit weight acquisition with mocked network; no torch/model dependency."""

import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deadline_engine import backend


class WeightSpecTests(unittest.TestCase):
    def libraries(self, filename):
        weights = SimpleNamespace(url="https://download.pytorch.org/models/" + filename)
        models = SimpleNamespace(MobileNet_V3_Small_Weights=SimpleNamespace(IMAGENET1K_V1=weights))
        return None, SimpleNamespace(models=models)

    def test_known_checkpoint_uses_full_pinned_sha256(self):
        with patch.object(backend, "_libraries", return_value=self.libraries("mobilenet_v3_small-047dcff4.pth")):
            spec = backend._weight_spec()
        self.assertEqual(spec[-1], backend.EXPECTED_WEIGHT_SHA256)
        self.assertEqual(len(spec[-1]), 64)

    def test_changed_checkpoint_requires_explicit_pin_review(self):
        with patch.object(backend, "_libraries", return_value=self.libraries("mobilenet_v3_small-aaaaaaaa.pth")):
            with self.assertRaisesRegex(RuntimeError, "unsupported.*weight URL"):
                backend._weight_spec()


class WeightDownloadTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".cache"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="test-download-", dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.checkpoints = Path(self.temporary.name) / "checkpoints"
        self.body = b"deterministic test checkpoint bytes"
        self.digest = hashlib.sha256(self.body).hexdigest()
        self.path = self.checkpoints / ("test-" + self.digest[:8] + ".pth")
        self.url = "https://example.invalid/" + self.path.name
        self.spec = (None, SimpleNamespace(__version__="test-version"),
                     SimpleNamespace(url=self.url), self.path, self.digest[:8])
        for replacement in (
            patch.object(backend, "CHECKPOINTS", self.checkpoints),
            patch.object(backend, "_weight_spec", return_value=self.spec),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)

    def assertCleanFailedDownload(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.checkpoints.iterdir()), [])

    def test_valid_cache_never_opens_network(self):
        self.checkpoints.mkdir()
        self.path.write_bytes(self.body)
        with patch.object(backend, "urlopen") as network:
            result = backend.fetch_weights()
        network.assert_not_called()
        self.assertEqual(result["sha256"], self.digest)
        self.assertEqual(result["expected_sha256"], self.digest[:8])
        self.assertEqual(result["torchvision_version"], "test-version")
        self.assertEqual(result["weights"], "MobileNet_V3_Small_Weights.IMAGENET1K_V1")

    def test_missing_cache_timeout_reports_error_and_cleans_temporary_file(self):
        with patch.object(backend, "urlopen", side_effect=TimeoutError("network timed out")) as network:
            with self.assertRaisesRegex(RuntimeError, "No randomly initialized model was substituted") as caught:
                backend.fetch_weights()
        network.assert_called_once_with(self.url, timeout=30)
        self.assertIn("network timed out", str(caught.exception))
        self.assertIn(self.url, str(caught.exception))
        self.assertCleanFailedDownload()

    def test_download_hash_mismatch_never_enters_cache(self):
        with patch.object(backend, "urlopen", return_value=io.BytesIO(b"tampered")):
            with self.assertRaisesRegex(RuntimeError, "SHA256 verification"):
                backend.fetch_weights()
        self.assertCleanFailedDownload()

    def test_corrupt_cache_is_preserved_for_explicit_removal_without_network(self):
        self.checkpoints.mkdir()
        self.path.write_bytes(b"damaged existing checkpoint")
        with patch.object(backend, "urlopen") as network:
            with self.assertRaisesRegex(RuntimeError, "Cached weights failed SHA256"):
                backend.fetch_weights()
        network.assert_not_called()
        self.assertEqual(self.path.read_bytes(), b"damaged existing checkpoint")

    def test_size_limit_rejects_oversized_response_and_cleans_up(self):
        with patch.object(backend, "MAX_WEIGHT_BYTES", 4), patch.object(
                backend, "urlopen", return_value=io.BytesIO(self.body)):
            with self.assertRaisesRegex(RuntimeError, "size limit"):
                backend.fetch_weights()
        self.assertCleanFailedDownload()

    def test_verified_download_is_cached_with_pinned_identity(self):
        with patch.object(backend, "urlopen", return_value=io.BytesIO(self.body)) as network:
            result = backend.fetch_weights()
        network.assert_called_once_with(self.url, timeout=30)
        self.assertEqual(self.path.read_bytes(), self.body)
        self.assertEqual(list(self.checkpoints.iterdir()), [self.path])
        self.assertEqual(result["sha256"], self.digest)
        self.assertEqual(result["url"], self.url)
        self.assertEqual(result["model"], "mobilenet_v3_small")
        self.assertEqual(result["torchvision_version"], "test-version")


if __name__ == "__main__":
    unittest.main()
