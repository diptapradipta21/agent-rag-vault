#!/usr/bin/env python3
"""
agent-rag-vault — Local Zero-Token RAG & Semantic Markdown Knowledge Vault for AI Agents.
Indexes local Obsidian vaults, Markdown notes, SOPs, and documentation using lightweight BM25 + n-gram keyword ranker.
Provides instant retrieval for Claude Code, Hermes, and local agents with 0 LLM token cost on queries.
"""

import os
import sys
import re
import math
import json
import argparse
from pathlib import Path
from collections import Counter

class MarkdownRAGVault:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path).resolve()
        self.documents = []  # List of {path, title, content, tokens, length}
        self.doc_freqs = Counter() # Term -> document count
        self.avg_doc_len = 0
        self.total_docs = 0

    def tokenize(self, text: str):
        """Clean and extract alphanumeric search tokens."""
        words = re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', text.lower())
        return words

    def index(self, max_files=2000, quiet=False):
        if not quiet:
            print(f"[*] Indexing Markdown vault: {self.vault_path}", file=sys.stderr)
        total_len = 0
        count = 0
        
        for p in self.vault_path.rglob("*.md"):
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
                    
                doc_entry = {
                    "rel_path": str(p.relative_to(self.vault_path)),
                    "abs_path": str(p),
                    "title": p.stem.replace("-", " ").replace("_", " ").title(),
                    "title_tokens": set(title_tokens),
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
        self.avg_doc_len = (total_len / self.total_docs) if self.total_docs > 0 else 0
        if not quiet:
            print(f"[✓] Indexed {self.total_docs} markdown notes (Avg tokens: {self.avg_doc_len:.1f}).", file=sys.stderr)

    def query(self, query_text: str, top_k: int = 5, k1: float = 1.5, b: float = 0.75, min_score: float = 0.0):
        """BM25 ranking query with title-match boost and 0 token cost."""
        q_tokens = self.tokenize(query_text)
        if not q_tokens or self.total_docs == 0:
            return []

        scores = []
        for doc in self.documents:
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
            results.append({
                "score": round(sc, 3),
                "title": doc["title"],
                "path": doc["rel_path"],
                "excerpt": excerpt
            })
        return results

def main():
    parser = argparse.ArgumentParser(description="Zero-Token Markdown Knowledge RAG Vault for AI Agents.")
    parser.add_argument("vault", help="Path to local Markdown or Obsidian vault directory")
    parser.add_argument("--query", "-q", required=True, help="Search query to retrieve relevant context notes")
    parser.add_argument("--limit", "-k", type=int, default=5, help="Number of matching documents to return (default: 5)")
    parser.add_argument("--min-score", "-m", type=float, default=0.0, help="Minimum BM25 score cutoff threshold (default: 0.0)")
    parser.add_argument("--json", action="store_true", help="Output results in clean JSON format for agent tool ingestion")
    parser.add_argument("--quiet", action="store_true", help="Suppress indexing progress output")

    args = parser.parse_args()

    vault = MarkdownRAGVault(args.vault)
    vault.index(quiet=(args.json or args.quiet))
    results = vault.query(args.query, top_k=args.limit, min_score=args.min_score)

    if args.json:
        print(json.dumps(results, indent=2))
        sys.exit(0)

    print(f"\n🔍 Top {len(results)} relevant notes for query: '{args.query}'\n")
    for idx, r in enumerate(results, 1):
        print(f"{idx}. [{r['score']}] {r['title']} ({r['path']})")
        print(f"   Excerpt: {r['excerpt']}\n")

if __name__ == "__main__":
    main()
