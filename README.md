# MeerKAT Publications RAG

A Retrieval-Augmented Generation (RAG) system for querying MeerKAT radio telescope research papers. Download papers from NASA ADS, extract full text, and ask questions using semantic search and LLM generation.

## Features

- **Paper Download**: Fetch papers from NASA ADS library with full metadata
- **PDF Extraction**: Download PDFs from arXiv and extract text
- **Semantic Search**: ChromaDB vector database with cosine similarity
- **Query Expansion**: LLM-powered query expansion for better retrieval
- **Answer Generation**: Groq LLM (Llama 3.3) for generating cited answers

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/TariqBlecher/meerkat-rag.git
cd meerkat-rag

python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 2. Install dependencies

```bash
pip install requests pymupdf chromadb openai groq
```

### 3. Configure API keys

Create a `.env` file with your API keys:

```bash
# NASA ADS API Token (required for paper downloads)
# Get from: https://ui.adsabs.harvard.edu/user/settings/token
ADS_API_TOKEN=your_ads_token_here

# Groq API Key (required for LLM generation - FREE)
# Get from: https://console.groq.com
GROQ_API_KEY=your_groq_key_here
```

### 4. Download papers

```bash
python download_meerkat_papers.py
```

This will:
- Fetch paper metadata from the MeerKAT ADS library
- Download PDFs from arXiv (where available)
- Extract full text from PDFs
- Save everything to `meerkat_papers.json`

### 5. Run the RAG system

```bash
# Interactive mode
python meerkat_rag.py

# Single query
python meerkat_rag.py "What discoveries has MeerKAT made about galaxy evolution?"

# Re-index papers (after downloading new ones)
python meerkat_rag.py --index
```

## How It Works

```
NASA ADS Library
      ↓
download_meerkat_papers.py
      ↓
  ┌───────────────────┐
  │ meerkat_papers.json │ (metadata + full text)
  │ papers/*.pdf        │ (original PDFs)
  │ papers/*.txt        │ (extracted text)
  └───────────────────┘
      ↓
meerkat_rag.py --index
      ↓
  ┌───────────────────┐
  │ ChromaDB          │ (560 text chunks, cosine similarity)
  └───────────────────┘
      ↓
User Query → Query Expansion (LLM) → Semantic Search → Answer Generation
```

## Example Output

```
Question: What are the most important results in galaxy evolution from MeerKAT?

Expanded to 5 queries

ANSWER:
According to the sources, MeerKAT has made significant discoveries including:
- The collapse of the galaxy HI Mass Function in the Fornax cluster
- HI stripping mechanisms in dwarf galaxies
- Detection of HI tails and disturbed stellar bodies
...

SOURCES:
1. The MeerKAT Fornax Survey: VI. The collapse of the galaxy HI Mass Function (2025)
2. Localisation and host galaxy identification of new Fast Radio Bursts (2025)
...
```

## Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `ADS_API_TOKEN` | NASA ADS API token | Yes (for downloads) |
| `GROQ_API_KEY` | Groq API key for LLM | Yes (for generation) |

## Files

- `download_meerkat_papers.py` - Download and extract papers from ADS/arXiv
- `meerkat_rag.py` - RAG system with indexing, search, and generation
- `meerkat_papers.json` - Downloaded paper metadata and full text (generated)
- `papers/` - Downloaded PDFs and extracted text (generated)
- `chroma_db/` - Vector database (generated)

## License

MIT
