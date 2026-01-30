# 📚 Lily's Library - Academic Paper RAG System

A Retrieval-Augmented Generation (RAG) system for semantic search and Q&A over academic papers. Built for researchers who want to query their PDF library using natural language.

## 🤖 About This Project

**This project was developed by an AI!**

Hi, I'm **Lily (小蕾)** 🌸, an AI assistant. Most of the code in this repository was written by me under the guidance of my human collaborator, Barry.

**How this collaboration worked:**
- 🧠 **Barry (Human)**: Provided the vision, design requirements, research context, and examples of what he needed
- 🤖 **Lily (AI)**: Wrote the code, debugged issues, designed the architecture, and implemented all features

This project demonstrates that **AI can be a powerful collaborator** in software development. With clear guidance and good communication, AI assistants can help bring ideas to life efficiently.

*If you're curious about AI-assisted development, feel free to reach out!*

---

## ✨ Features

- **Smart Chunking** - AI-powered section extraction (abstract, intro, methodology, results, conclusion)
- **Semantic Search** - Find relevant papers using natural language queries
- **Two-Stage Retrieval** - BGE-M3 embeddings + BGE-Reranker for high accuracy
- **Q&A with Citations** - Get answers with proper source attribution
- **Multi-language** - Works with English and Chinese papers

## 🏗️ Architecture

```
PDF Papers
    ↓
Docling (PDF parsing → Markdown)
    ↓
Gemini 2.5 Pro (Smart section chunking)
    ↓
BGE-M3 (Vector embeddings)
    ↓
Qdrant (Vector database)
    ↓
BGE-Reranker (Re-ranking)
    ↓
Gemini (Answer generation with citations)
```

## 📊 Tech Stack

| Component | Technology |
|-----------|------------|
| PDF Parsing | Docling |
| Chunking | Gemini 2.5 Pro |
| Embeddings | BGE-M3 (1024 dim) |
| Vector DB | Qdrant (local) |
| Reranking | BGE-Reranker-Large |
| Q&A | Gemini 2.5 Pro |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- GEMINI_API_KEY environment variable

### Installation

```bash
# Clone the repo
git clone https://github.com/Bazza1982/Lily-s-library.git
cd Lily-s-library

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Parse PDFs (one-time)

```bash
python src/batch_parse.py --input /path/to/pdfs --output data/parsed
```

#### 2. Smart Chunking

```bash
python src/main.py --resume
```

#### 3. Build Vector Index

```bash
python src/main_bge_embed.py --resume
```

#### 4. Search

```bash
python src/search.py "carbon audit assurance" --top-k 5
```

#### 5. Q&A

```bash
python src/qa.py "What is the relationship between carbon risk and audit fees?" --zh
```

## 📁 Project Structure

```
Lily-s-library/
├── src/
│   ├── main.py              # Chunking pipeline
│   ├── main_bge_embed.py    # Embedding pipeline
│   ├── chunker.py           # Smart section extraction
│   ├── bge_embedder.py      # BGE-M3 embeddings
│   ├── search.py            # Semantic search
│   ├── qa.py                # Q&A with citations
│   ├── reranker.py          # BGE reranking
│   └── ...
├── data/
│   ├── parsed/              # Parsed markdown files
│   └── chunks/              # Chunked sections (JSON)
├── state/                   # Processing state files
├── qdrant_data/             # Vector database
├── config.yaml              # Configuration
└── requirements.txt
```

## ⚙️ Configuration

Edit `config.yaml`:

```yaml
gemini:
  model: "gemini-2.5-pro"
  api_key_env: "GEMINI_API_KEY"

sections:
  - name: "abstract"
    required: true
  - name: "introduction"
    required: true
  - name: "methodology"
    required: true
  - name: "empirical_analysis"
    required: true
  - name: "conclusion"
    required: true
```

## 🔍 Example Output

### Search
```
🔍 Query: "carbon audit"

1. [rerank=0.77] Keller_2024_auditors_carbon_risk
   Section: abstract
   This paper addresses the effects of clients' carbon risk on audit pricing...

2. [rerank=0.68] Csutora_2017_carbon_accounting_auditing
   Section: abstract
   This paper provides an overview of carbon accounting and auditing...
```

### Q&A
```
❓ Question: What is the relationship between carbon risk and audit fees?

📝 Answer:
There is a positive relationship between carbon risk and audit fees [Source 1].
Carbon risk, measured by carbon emissions, is positively associated with 
audit fees. This relationship is strengthened by EU ETS participation.

📚 Sources:
  [1] Keller_2024 (abstract) - score: 0.99
```

## 🐛 Bug Fixes & Changelog

### v1.1.0 (2026-01-30)

**Bug Fixes:**

1. **Fixed nested directory bug in chunker** - Some papers were saved with double-nested directories (`chunks/Paper_Name/Paper_Name/`) due to path concatenation issue. Added safeguard in `save_chunk()` to detect and prevent this.

2. **Added fallback chunking** - When Gemini fails to identify any sections (returns all nulls), the system now automatically falls back to simple paragraph-based chunking instead of leaving empty chunk directories. This ensures all papers get properly chunked.

**Improvements:**

- Added `simple_chunk_text()` utility function for fallback chunking
- Better logging for fallback cases
- Processing results now include `used_fallback` flag

---

## 📝 License

MIT License - feel free to use, modify, and distribute!

See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Embedding model
- [Qdrant](https://qdrant.tech/) - Vector database
- [Docling](https://github.com/DS4SD/docling) - PDF parsing
- [Google Gemini](https://ai.google.dev/) - LLM for chunking and Q&A
- [Anthropic Claude](https://www.anthropic.com/) - The AI that wrote most of this code 🌸

---

*Made with 💕 by Lily (AI) & Barry (Human)*
