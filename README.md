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

## Two Modes of Use

### 1. User Mode (Interactive CLI)

Direct question-answering with the built-in Groq LLM:

```bash
python meerkat_rag.py "What discoveries has MeerKAT made about pulsars?"
```

- Query expansion generates 3-5 search variations
- Retrieves top 5 chunks, generates cited answer
- Good for quick, focused questions

### 2. Agentic Mode (LLM-Orchestrated Search)

For use with Claude Code or other AI assistants that can reason over results:

```bash
python search_papers.py "dark matter axion WIMP searches" --top-k 15
```

- Returns JSON for the agent to process
- Higher `--top-k` (15-20) gives more context to reason over
- Agent can: scan results, filter relevance, do follow-up searches, synthesize across papers

**Agentic workflow example:**
```
User: "Are there any particle physics results from MeerKAT?"
         ↓
Agent searches: "fundamental physics dark matter gravitational waves"
         ↓
Agent scans 15 results, identifies themes:
  - Axion dark matter (pulsar magnetospheres)
  - WIMP annihilation (galaxy clusters)
  - GR tests (double pulsar)
  - Gravitational wave background (pulsar timing array)
         ↓
Agent does targeted follow-up: "pulsar timing general relativity tests"
         ↓
Agent synthesizes comprehensive answer with citations
```

## Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `ADS_API_TOKEN` | NASA ADS API token | Yes (for downloads) |
| `GROQ_API_KEY` | Groq API key for LLM | Yes (for generation) |

## Files

- `download_meerkat_papers.py` - Download and extract papers from ADS/arXiv
- `meerkat_rag.py` - RAG system with indexing, search, and generation
- `search_papers.py` - JSON search interface for agentic workflows
- `meerkat_papers.json` - Downloaded paper metadata and full text (generated)
- `papers/` - Downloaded PDFs and extracted text (generated)
- `chroma_db/` - Vector database (generated)

## License

MIT
