# 📚 Academic Paper RAG System

A Retrieval-Augmented Generation (RAG) system for semantic search and Q&A over academic papers. Built for researchers who want to query their PDF library using natural language.

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
git clone https://github.com/yourusername/academic-paper-rag.git
cd academic-paper-rag

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
academic-paper-rag/
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

## 📝 License

MIT License - feel free to use for your research!

## 🙏 Acknowledgments

- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Embedding model
- [Qdrant](https://qdrant.tech/) - Vector database
- [Docling](https://github.com/DS4SD/docling) - PDF parsing
- [Google Gemini](https://ai.google.dev/) - LLM for chunking and Q&A
