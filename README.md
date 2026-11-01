# 🗄️ Agent RAG Vault

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Zero Token Cost](https://img.shields.io/badge/RAG%20Query%20Cost-0%20Tokens-green.svg)](#)
[![Obsidian Ready](https://img.shields.io/badge/Obsidian-Compatible-purple.svg)](https://obsidian.md/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Local Zero-Token RAG & Information Retrieval Engine for AI Agents. Indexes local Obsidian vaults, SOPs, and Markdown knowledge bases using BM25 ranking — instant retrieval without vector DB costs.**

---

## 💡 The Problem
AI agents (Claude Code, Hermes, AutoGen) frequently suffer from **context amnesia** or burn thousands of dollars querying expensive cloud Vector DBs (Pinecone, Chroma, Qdrant) just to find simple internal notes, business SOPs, or past decisions.

## 🚀 The Solution
`agent-rag-vault` delivers a blazingly fast, zero-dependency, local BM25 search engine that reads directly from your **Obsidian Vault** or local markdown folders:
1. **0 Extra API Tokens:** Queries run 100% locally on CPU in under 50ms.
2. **JSON Output for LLM Ingestion:** Easily piped into Claude tools, MCP servers, or bash pre-hooks.
3. **No Heavy Embeddings:** No PyTorch, no HuggingFace downloads, no disk-heavy vector indices.

---

## ⚡ Quick Start

### 1. Search Your Knowledge Vault
```bash
python3 vault_rag.py /path/to/obsidian-vault --query "Meta Ads daily budget rules"
```

### 2. Output as Structured JSON for AI Agents
```bash
python3 vault_rag.py /path/to/obsidian-vault -q "SOP Nazhir Wakaf" --json
```

### 3. High-Confidence Search with Score Threshold
```bash
# Filter out low-relevance results using --min-score
python3 vault_rag.py /path/to/obsidian-vault -q "Yoast SEO rules" --min-score 2.5
```

---

## 🛠️ CLI Flags
- `--query, -q`: Search terms to rank against indexed notes.
- `--limit, -k`: Max items to return (default: 5).
- `--min-score, -m`: Minimum BM25 cutoff threshold (default: 0.0).
- `--json`: Output clean JSON array (progress logs automatically redirected to `stderr`).
- `--quiet`: Suppress indexing progress output.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
