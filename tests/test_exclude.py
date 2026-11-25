import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vault_rag import MarkdownRAGVault


class TestExcludePatterns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vault_path = Path(self.tmp) / "vault"
        self.vault_path.mkdir()
        (self.vault_path / "keep.md").write_text("Alpha deployment checklist.", encoding="utf-8")
        arch = self.vault_path / "archive"
        arch.mkdir()
        (arch / "old.md").write_text("Old alpha note.", encoding="utf-8")
        (self.vault_path / "draft.tmp.md").write_text("Temporary alpha draft.", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_exclude_glob_patterns(self):
        vault = MarkdownRAGVault(str(self.vault_path), exclude=["archive/*", "*.tmp.md"])
        vault.index(quiet=True)
        self.assertEqual(vault.total_docs, 1)
        self.assertEqual([d["rel_path"] for d in vault.documents], ["keep.md"])

    def test_exclude_matches_directory_segment(self):
        vault = MarkdownRAGVault(str(self.vault_path), exclude=["archive"])
        vault.index(quiet=True)
        self.assertEqual([d["rel_path"] for d in vault.documents], ["keep.md", "draft.tmp.md"])

    def test_exclude_empty_list_indexes_all(self):
        vault = MarkdownRAGVault(str(self.vault_path), exclude=[])
        vault.index(quiet=True)
        self.assertEqual(vault.total_docs, 3)


if __name__ == "__main__":
    unittest.main()
