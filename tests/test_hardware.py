r"""Real pretrained model checks. Explicit opt-in; not part of bounded CPU CI.

PowerShell: $env:RUN_GPU_TESTS='1'; .venv\Scripts\python.exe -m unittest discover -s tests -p test_hardware.py -v
For real CPU model checks, set RUN_MODEL_TESTS=1 instead (or as well).
Fetch weights explicitly first. An unavailable requested GPU is a test failure.
"""

import os
import unittest
from types import SimpleNamespace

from deadline_engine.backend import TorchBackend


@unittest.skipUnless(os.environ.get("RUN_MODEL_TESTS") == "1", "set RUN_MODEL_TESTS=1 for cached CPU model tests")
class PretrainedCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = TorchBackend(device="cpu", max_batch=2, seed=7)
        cls.backend.prepare()

    def test_cached_pretrained_cpu_predictions_and_accounting_metadata(self):
        requests = [SimpleNamespace(input_index=i) for i in range(2)]
        single = [self.backend.infer([request])["predictions"][0] for request in requests]
        batched = self.backend.infer(requests)
        self.assertEqual(batched["predictions"], single)
        self.assertGreater(batched["device_ms"], 0)
        self.assertGreaterEqual(batched["service_ms"], batched["device_ms"])
        metadata = self.backend.metadata()
        self.assertEqual(metadata["device"], "cpu")
        self.assertEqual(metadata["weights"], "MobileNet_V3_Small_Weights.IMAGENET1K_V1")
        self.assertIsNone(self.backend.peak_memory()["cuda_peak_allocated_bytes"])
        self.assertEqual(set(metadata["calibration"]), {"1", "2"})

    def test_unsupported_input_metadata_fails_explicitly(self):
        for request in (
            SimpleNamespace(input_index=0, shape=(3, 112, 112), dtype="float32"),
            SimpleNamespace(input_index=0, shape=(3, 224, 224), dtype="float16"),
            SimpleNamespace(input_index=True),
            SimpleNamespace(input_index=-1),
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.backend.infer([request])


@unittest.skipUnless(os.environ.get("RUN_GPU_TESTS") == "1", "set RUN_GPU_TESTS=1 on a CUDA host")
class PretrainedCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = TorchBackend(device="cuda", max_batch=4, seed=7)
        cls.backend.prepare()

    def test_batch_matches_singleton_predictions_and_forward_values(self):
        """Batching changes grouping, while these exact inputs stay the same."""
        backend = self.backend
        requests = [SimpleNamespace(input_index=i) for i in range(4)]
        single = [backend.infer([request])["predictions"][0] for request in requests]
        batched = backend.infer(requests)
        self.assertEqual(batched["predictions"], single)
        self.assertEqual(len(batched["predictions"]), 4)
        self.assertGreater(batched["device_ms"], 0)
        self.assertGreaterEqual(batched["service_ms"], batched["device_ms"])

        # Inspect logits too: equal argmax alone could conceal a numerical error.
        # This test alone uses private tensors; product callers use infer().
        torch = backend._torch
        with torch.inference_mode():
            images = backend._pool[:4].to("cuda")
            grouped = backend._model(images)
            separate = torch.cat([backend._model(image.unsqueeze(0)) for image in images])
            torch.cuda.synchronize()
        self.assertEqual(tuple(grouped.shape), (4, 1000))
        self.assertTrue(torch.isfinite(grouped).all().item())
        torch.testing.assert_close(grouped, separate, rtol=1e-3, atol=1e-4)

    def test_repeat_prepare_preserves_shared_calibration_and_resets_stats(self):
        before = dict(self.backend.estimates_s)
        count = self.backend.metadata()["prepare_count"]
        self.backend.prepare()
        self.assertEqual(self.backend.estimates_s, before)
        self.assertEqual(self.backend.metadata()["prepare_count"], count + 1)
        memory = self.backend.peak_memory()
        self.assertGreater(memory["cuda_peak_allocated_bytes"], 0)
        self.assertGreaterEqual(memory["cuda_peak_reserved_bytes"], memory["cuda_peak_allocated_bytes"])
        self.assertEqual(self.backend.metadata()["device"], "cuda")
        self.assertEqual(self.backend.metadata()["weights"], "MobileNet_V3_Small_Weights.IMAGENET1K_V1")


if __name__ == "__main__":
    unittest.main()
