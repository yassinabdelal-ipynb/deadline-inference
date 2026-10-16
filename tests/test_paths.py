import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from deadline_engine.paths import get_cache_root, get_data_root, project_root


class PortablePathTests(unittest.TestCase):
    def test_source_cache_remains_local(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_cache_root(), project_root()/".cache")

    def test_explicit_data_override_is_used_for_cache(self):
        value = str(project_root()/".cache"/"portable-test")
        with patch.dict(os.environ, {"DEADLINE_ENGINE_DATA_DIR":value}):
            self.assertEqual(get_data_root(), Path(value).resolve())
            self.assertEqual(get_cache_root(), Path(value).resolve()/"cache")

    def test_frozen_app_writes_to_user_data_not_executable_folder(self):
        location = str(project_root()/".cache"/"user-profile-test")
        with patch.dict(os.environ, {"LOCALAPPDATA": location, "HOME": location}, clear=True), patch.object(sys,"frozen",True,create=True):
            self.assertEqual(get_cache_root(), get_data_root()/"cache")
            self.assertNotEqual(get_cache_root(), project_root()/".cache")
