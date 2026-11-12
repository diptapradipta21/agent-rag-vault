#!/usr/bin/env python3
"""
agent-rag-vault — Local Zero-Token RAG & Semantic Markdown Knowledge Vault for AI Agents.
Indexes local Obsidian vaults, Markdown notes, SOPs, and documentation using lightweight BM25 + n-gram keyword ranker.
Provides instant retrieval for Claude Code, Hermes, and local agents with 0 LLM token cost on queries.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set


class MarkdownRAGVault:
    def __init__(self, vault_path: str, extensions: Optional[List[str]] = None):
        self.vault_path = Path(vault_path).resolve()
        self.extensions = set(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in (extensions or [".md", ".markdown"]))
        self.documents: List[Dict[str, Any]] = []
        self.doc_freqs: Counter = Counter()
        self.avg_doc_len = 0.0
        self.total_docs = 0

    def tokenize(self, text: Optional[str]) -> List[str]:
        """Clean and extract alphanumeric search tokens."""
        if not text:
            return []
        return re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', text.lower())

    def extract_tags(self, content: str) -> Set[str]:
        """Extract tags from YAML frontmatter and inline markdown #hashtags."""
        tags: Set[str] = set()

        # Frontmatter tags: YAML block between first pair of ---
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                fm = parts[1]
                # Look for 'tags: [a, b]' or 'tags: \n - a'
                fm_match = re.search(r'(?m)^tags:\s*(.*)$', fm)
                if fm_match:
                    inline_tags = fm_match.group(1).strip()
                    if inline_tags.startswith("[") and inline_tags.endswith("]"):
                        raw_list = inline_tags[1:-1].split(",")
                        for t in raw_list:
                            clean = t.strip().strip("'\"").lstrip("#").lower()
                            if clean:
                                tags.add(clean)
                    elif not inline_tags:
                        # Multi-line list items
                        for line in fm[fm_match.end():].splitlines():
                            line_stripped = line.strip()
                            if line_stripped.startswith("- "):
                                val = line_stripped[2:].strip().strip("'\"").lstrip("#").lower()
                                if val:
                                    tags.add(val)
                            elif line.startswith(" ") or line.startswith("\t"):
                                continue
                            else:
                                break

        # Inline hashtags: #tag (must not be markdown header like '# Header')
        # Matches #tag, #sub_tag, #sub-tag where preceded by start or whitespace
        for match in re.finditer(r'(?:^|\s)#([a-zA-Z0-9_\-\/]{2,})', content):
            tag_name = match.group(1).strip().lower()
            if not tag_name.isdigit():
                tags.add(tag_name)

        return tags

    def index(self, max_files: int = 2000, quiet: bool = False) -> None:
        if not quiet:
            print(f"[*] Indexing Markdown vault: {self.vault_path}", file=sys.stderr)
        total_len = 0
        count = 0

        for p in self.vault_path.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in self.extensions:
                continue
            # Exclude hidden files, git, and node_modules
            if any(part.startswith(".") for part in p.parts):
                continue
            if count >= max_files:
                break

            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    continue
                title_tokens = self.tokenize(p.stem)
                content_tokens = self.tokenize(content)
                tokens = title_tokens + content_tokens
                if not tokens:
                    continue

                tags = self.extract_tags(content)

                doc_entry: Dict[str, Any] = {
                    "rel_path": str(p.relative_to(self.vault_path)),
                    "abs_path": str(p),
                    "title": p.stem.replace("-", " ").replace("_", " ").title(),
                    "title_tokens": set(title_tokens),
                    "tags": tags,
                    "content": content,
                    "tokens": tokens,
                    "length": len(tokens)
                }

                # Update document frequencies (unique terms per doc)
                unique_terms = set(tokens)
                for term in unique_terms:
                    self.doc_freqs[term] += 1

                total_len += len(tokens)
                self.documents.append(doc_entry)
                count += 1
            except Exception:
                continue

        self.total_docs = len(self.documents)
        self.avg_doc_len = (total_len / self.total_docs) if self.total_docs > 0 else 0.0
        if not quiet:
            print(f"[✓] Indexed {self.total_docs} notes (Avg tokens: {self.avg_doc_len:.1f}).", file=sys.stderr)

    def list_tags(self) -> Dict[str, int]:
        """Aggregate and count frequency of all tags across indexed documents."""
        tag_counts: Counter = Counter()
        for doc in self.documents:
            for t in doc.get("tags", set()):
                tag_counts[t] += 1
        return dict(sorted(tag_counts.items(), key=lambda x: (-x[1], x[0])))

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
        min_score: float = 0.0,
        tag: Optional[str] = None,
        include_content: bool = False
    ) -> List[Dict[str, Any]]:
        """BM25 ranking query with title-match boost, tag filtering, and 0 token cost."""
        q_tokens = self.tokenize(query_text)
        if not q_tokens or self.total_docs == 0:
            return []

        clean_tag = tag.lstrip("#").lower() if tag else None

        scores = []
        for doc in self.documents:
            if clean_tag and clean_tag not in doc.get("tags", set()):
                continue

            score = 0.0
            doc_len = doc["length"]
            term_counts = Counter(doc["tokens"])
            title_tokens = doc.get("title_tokens", set())

            for term in q_tokens:
                if term not in self.doc_freqs:
                    continue
                df = self.doc_freqs[term]
                # Standard BM25 IDF
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf = term_counts.get(term, 0)
                # BM25 TF weight
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len))
                term_score = idf * (numerator / denominator)

                # Title-match boost: 1.5x bonus if query term is in doc title
                if term in title_tokens:
                    term_score *= 1.5

                score += term_score

            if score > min_score:
                scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc, sc in scores[:top_k]:
            # Extract high-relevance excerpt
            lines = [l.strip() for l in doc["content"].split("\n") if l.strip() and not l.startswith("#")]
            excerpt = " ".join(lines[:3])[:280] + "..." if lines else doc["content"][:280]
            entry: Dict[str, Any] = {
                "score": round(sc, 3),
                "title": doc["title"],
                "path": doc["rel_path"],
                "tags": sorted(list(doc.get("tags", set()))),
                "excerpt": excerpt
            }
            if include_content:
                entry["content"] = doc["content"]
            results.append(entry)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-Token Markdown Knowledge RAG Vault for AI Agents.")
    parser.add_argument("vault", help="Path to local Markdown or Obsidian vault directory")
    parser.add_argument("--query", "-q", default=None, help="Search query to retrieve relevant context notes")
    parser.add_argument("--list-tags", action="store_true", help="List all unique tags indexed across the vault with note counts")
    parser.add_argument("--full", action="store_true", help="Include complete note markdown content in output")
    parser.add_argument("--limit", "-k", type=int, default=5, help="Number of matching documents to return (default: 5)")
    parser.add_argument("--min-score", "-m", type=float, default=0.0, help="Minimum BM25 score cutoff threshold (default: 0.0)")
    parser.add_argument("--tag", "-t", help="Filter results by tag (e.g. 'sop' or '#finance')")
    parser.add_argument("--extensions", default=".md,.markdown", help="Comma-separated file extensions to index (default: .md,.markdown)")
    parser.add_argument("--json", action="store_true", help="Output results in clean JSON format for agent tool ingestion")
    parser.add_argument("--quiet", action="store_true", help="Suppress indexing progress output")

    args = parser.parse_args()

    ext_list = [e.strip() for e in args.extensions.split(",") if e.strip()]
    vault = MarkdownRAGVault(args.vault, extensions=ext_list)
    vault.index(quiet=(args.json or args.quiet))
    if args.list_tags:
        tags = vault.list_tags()
        if args.json:
            print(json.dumps(tags, indent=2))
        else:
            print(f"\n🏷️  Indexed Tags ({len(tags)} unique):\n")
            for t, count in tags.items():
                print(f"  • #{t}: {count} note(s)")
            print()
        sys.exit(0)

    if not args.query:
        print("[!] Error: Either --query or --list-tags is required.", file=sys.stderr)
        sys.exit(1)

    results = vault.query(
        args.query,
        top_k=args.limit,
        min_score=args.min_score,
        tag=args.tag,
        include_content=args.full
    )

    if args.json:
        print(json.dumps(results, indent=2))
        sys.exit(0)

    print(f"\n🔍 Top {len(results)} relevant notes for query: '{args.query}'\n")
    for idx, r in enumerate(results, 1):
        tag_str = f" [{', '.join(r['tags'])}]" if r.get("tags") else ""
        print(f"{idx}. [{r['score']}] {r['title']} ({r['path']}){tag_str}")
        print(f"   Excerpt: {r['excerpt']}")
        if args.full and "content" in r:
            print(f"\n--- Content ---\n{r['content']}\n---------------")
        print()
if __name__ == "__main__":
    main()
