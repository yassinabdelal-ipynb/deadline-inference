"""Publication boundaries: unsafe paths, private data and bulk stay out of ZIPs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
from build_release import source_allowed
import check_release
sys.path.remove(str(SCRIPTS))


class PublicationTests(unittest.TestCase):
    def test_source_allowlist_keeps_only_curated_results(self):
        self.assertTrue(source_allowed(Path("results/image-models/resnet18/cpu/profile/profile.csv")))
        self.assertTrue(source_allowed(Path("docs/public/results.json")))
        self.assertTrue(source_allowed(Path(".github/workflows/ci.yml")))
        for path in (".venv/lib/torch.py", "docs/.private/token.json", "docs/ENGINEERING_LOG.md",
                     "results/environment.json", "results/image-models/resnet18/cpu/comparison/requests.csv",
                     "results/benchmark/comparison.csv", "scripts/__pycache__/module.pyc", "docs/model.pth"):
            with self.subTest(path=path):
                self.assertFalse(source_allowed(Path(path)))

    def test_archive_paths_cannot_escape_or_include_bulk(self):
        for name in ("../secret", "/root/file", "C:/absolute/file", "safe\\..\\file", "safe/.venv/file",
                     "safe/.env", "safe/weights.pth", "safe/private.key"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                check_release.safe_member(name)
        self.assertEqual(str(check_release.safe_member("DeadlineInference/_internal/torch.dll")), "DeadlineInference/_internal/torch.dll")

    def test_source_text_scan_detects_values_without_echoing_them(self):
        token = "gh" + "p_" + "a"*36
        with self.assertRaises(ValueError) as raised:
            check_release.check_text("example.py", ("token="+token).encode())
        self.assertNotIn(token, str(raised.exception))
        personal = "C:" + "\\Users\\" + "private-owner" + "\\file.txt"
        with self.assertRaisesRegex(ValueError, "Personal home"):
            check_release.check_text("example.md", personal.encode())
        check_release.check_text("workflow.yml", b"GH_TOKEN: ${{ github.token }}")

    def test_archive_checksum_and_contents_are_both_required(self):
        scratch = SCRIPTS.parent/".cache"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="release test with spaces ", dir=scratch) as folder:
            root = Path(folder)
            archive = root/"deadline-inference-source.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                for name in ("README.md", "LICENSE", "deadline_engine/core.py"):
                    handle.writestr("deadline-inference/"+name, "test")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            manifest = root/"SHA256SUMS.txt"
            manifest.write_text(digest+"  "+archive.name+"\n", encoding="utf-8")
            self.assertEqual(len(check_release.check_archives(root)), 1)
            manifest.write_text("0"*64+"  "+archive.name+"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                check_release.check_archives(root)


if __name__ == "__main__":
    unittest.main()
