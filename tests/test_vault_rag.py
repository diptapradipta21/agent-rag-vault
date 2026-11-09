import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vault_rag import MarkdownRAGVault


class TestMarkdownRAGVault(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.vault_path = Path(self.test_dir) / "vault"
        self.vault_path.mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_tokenize(self):
        vault = MarkdownRAGVault(str(self.vault_path))
        tokens = vault.tokenize("Hello World! Quick brown fox 123_456.")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("123_456", tokens)

    def test_indexing_and_query(self):
        doc1 = self.vault_path / "sop-deployment.md"
        doc1.write_text("# Deployment SOP\nFollow these steps to deploy production docker swarm.", encoding="utf-8")

        doc2 = self.vault_path / "marketing-playbook.md"
        doc2.write_text("# Meta Ads Playbook\nOptimizing ROAS, CTR, and lookalike audiences.", encoding="utf-8")

        vault = MarkdownRAGVault(str(self.vault_path))
        vault.index(quiet=True)
        self.assertEqual(vault.total_docs, 2)

        results = vault.query("docker deployment", top_k=2)
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["title"], "Sop Deployment")
        self.assertIn("sop-deployment.md", results[0]["path"])

    def test_title_boost_and_score_cutoff(self):
        doc1 = self.vault_path / "kubernetes.md"
        doc1.write_text("General guide on cluster setup.", encoding="utf-8")

        doc2 = self.vault_path / "general.md"
        doc2.write_text("Mentioning kubernetes once here.", encoding="utf-8")

        vault = MarkdownRAGVault(str(self.vault_path))
        vault.index(quiet=True)

        results = vault.query("kubernetes", top_k=5)
        self.assertTrue(len(results) >= 2)
        self.assertEqual(results[0]["title"], "Kubernetes")

        strict_results = vault.query("kubernetes", top_k=5, min_score=999.0)
        self.assertEqual(len(strict_results), 0)

    def test_tag_extraction_and_filtering(self):
        doc1 = self.vault_path / "backend-ops.md"
        doc1.write_text("---\ntags: [devops, infra]\n---\n# Server Setup\nManaging docker containers.", encoding="utf-8")

        doc2 = self.vault_path / "billing-guide.md"
        doc2.write_text("# Stripe Billing\nHandling recurring payments #finance #ops", encoding="utf-8")

        vault = MarkdownRAGVault(str(self.vault_path))
        vault.index(quiet=True)

        # Both match generic query
        all_results = vault.query("setup payments", top_k=5)
        self.assertEqual(len(all_results), 2)

        # Filter by tag
        devops_results = vault.query("setup", tag="devops")
        self.assertEqual(len(devops_results), 1)
        self.assertEqual(devops_results[0]["title"], "Backend Ops")
        self.assertIn("devops", devops_results[0]["tags"])

        finance_results = vault.query("payments", tag="#finance")
        self.assertEqual(len(finance_results), 1)
        self.assertEqual(finance_results[0]["title"], "Billing Guide")
        self.assertIn("finance", finance_results[0]["tags"])


if __name__ == "__main__":
    unittest.main()
