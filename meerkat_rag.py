#!/usr/bin/env python3
"""
MeerKAT Publications RAG System
Query MeerKAT telescope research papers using retrieval-augmented generation.

Usage:
    python meerkat_rag.py                    # Interactive mode
    python meerkat_rag.py --index            # Re-index papers
    python meerkat_rag.py "your question"    # Single query
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# Load environment variables from .env file
def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value

load_env()

# Configuration
PAPERS_FILE = Path(__file__).parent / "meerkat_papers.json"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "meerkat_papers"

CHUNK_SIZE = 500  # words per chunk
CHUNK_OVERLAP = 50  # overlapping words


@dataclass
class SearchResult:
    """A single search result from the RAG system."""
    text: str
    title: str
    authors: str
    year: str
    arxiv_id: Optional[str]
    similarity: float

    @property
    def arxiv_url(self) -> Optional[str]:
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        return None


class MeerKATRAG:
    """RAG system for MeerKAT publications."""

    def __init__(self, persist_dir: str = None):
        """Initialize the RAG system."""
        self.persist_dir = persist_dir or str(CHROMA_DIR)

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection (uses built-in embeddings)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize Groq client for LLM
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            print("Warning: GROQ_API_KEY not set. Generation will not work.")
            self.llm = None
        else:
            self.llm = OpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1"
            )

        print(f"Initialized RAG system")
        print(f"  Collection: {self.collection.count()} chunks indexed")

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            if len(chunk.split()) >= 50:  # Skip very small chunks
                chunks.append(chunk)

        return chunks

    def expand_query(self, query: str) -> List[str]:
        """Expand a query into multiple variations for better retrieval."""
        if not self.llm:
            return [query]

        try:
            response = self.llm.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a query expansion system for a RAG database of MeerKAT radio telescope papers.

Your task: Generate search query variations optimized for semantic similarity matching against academic papers.

Guidelines:
- Add relevant technical synonyms and related terms
- Include common acronyms and their expansions (MIGHTEE, LADUMA, MALS, MeerTRAP, etc.)
- Add domain-specific terminology (flux density, spectral line, HI 21cm, continuum, etc.)
- Focus on terms that would actually appear in paper abstracts/content
- Return ONLY a valid JSON array of 3-5 query strings, nothing else

Examples:
User: "How sensitive is MeerKAT?"
Output: ["MeerKAT sensitivity noise rms detection limit", "MeerKAT flux density Jy/beam continuum spectral line sensitivity", "MeerKAT telescope performance noise characteristics"]

User: "What surveys use MeerKAT?"
Output: ["MeerKAT surveys MIGHTEE LADUMA MALS MeerTRAP", "MeerKAT large survey programs observing deep field", "MeerKAT sky survey radio continuum HI observations"]"""
                    },
                    {
                        "role": "user",
                        "content": f'Expand this query: "{query}"'
                    }
                ],
                temperature=0.7,
                max_tokens=400
            )

            content = response.choices[0].message.content.strip()
            # Parse JSON array
            expanded = json.loads(content)
            if isinstance(expanded, list) and len(expanded) > 0:
                # Always include original query
                if query not in expanded:
                    expanded.insert(0, query)
                return expanded[:5]  # Limit to 5 queries
        except Exception as e:
            print(f"  Query expansion failed: {e}")

        return [query]

    def index_papers(self, papers_file: str = None):
        """Index papers into ChromaDB."""
        papers_path = Path(papers_file) if papers_file else PAPERS_FILE

        with open(papers_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        papers = data['papers']
        print(f"Indexing {len(papers)} papers...")

        # Clear existing collection
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        all_chunks = []
        all_metadata = []
        all_ids = []

        for paper in papers:
            # Use full text if available, otherwise abstract
            text = paper.get('full_text') or paper.get('abstract', '')
            if not text:
                continue

            title = paper.get('title', 'Unknown')
            authors = paper.get('authors', [])
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            # Chunk the text
            chunks = self.chunk_text(text)

            for idx, chunk in enumerate(chunks):
                chunk_id = f"{paper.get('bibcode', 'unknown')}_{idx}"

                metadata = {
                    "title": (title or "Unknown")[:500],  # ChromaDB has metadata limits
                    "authors": (author_str or "")[:200],
                    "year": str(paper.get('year') or ''),
                    "arxiv_id": paper.get('arxiv_id') or '',
                    "bibcode": paper.get('bibcode') or '',
                    "chunk_idx": idx,
                    "total_chunks": len(chunks)
                }

                all_chunks.append(chunk)
                all_metadata.append(metadata)
                all_ids.append(chunk_id)

        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            end = min(i + batch_size, len(all_chunks))
            self.collection.add(
                documents=all_chunks[i:end],
                metadatas=all_metadata[i:end],
                ids=all_ids[i:end]
            )
            print(f"  Indexed {end}/{len(all_chunks)} chunks...")

        print(f"Indexing complete: {len(all_chunks)} chunks from {len(papers)} papers")

    def search(self, query: str, top_k: int = 5, expand: bool = True) -> List[SearchResult]:
        """Search for relevant chunks with optional query expansion."""
        # Expand query for better retrieval
        if expand:
            queries = self.expand_query(query)
            print(f"  Expanded to {len(queries)} queries")
        else:
            queries = [query]

        # Search with all query variations
        all_results = {}  # Use dict to deduplicate by chunk ID

        for q in queries:
            results = self.collection.query(
                query_texts=[q],
                n_results=top_k
            )

            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                chunk_id = results['ids'][0][i]

                # Keep best similarity score for duplicates
                similarity = 1 - distance
                if chunk_id not in all_results or all_results[chunk_id].similarity < similarity:
                    all_results[chunk_id] = SearchResult(
                        text=doc,
                        title=meta.get('title', 'Unknown'),
                        authors=meta.get('authors', ''),
                        year=meta.get('year', ''),
                        arxiv_id=meta.get('arxiv_id') or None,
                        similarity=similarity
                    )

        # Sort by similarity and return top_k
        search_results = sorted(all_results.values(), key=lambda x: x.similarity, reverse=True)
        return search_results[:top_k]

    def generate_answer(self, query: str, top_k: int = 5, sort_by_year: bool = False) -> Dict:
        """Generate an answer using RAG."""
        if not self.llm:
            return {
                "answer": "Error: GROQ_API_KEY not configured",
                "sources": []
            }

        # Retrieve relevant chunks
        results = self.search(query, top_k=top_k)

        if not results:
            return {
                "answer": "No relevant papers found for this query.",
                "sources": []
            }

        # Optionally sort by year (chronologically)
        if sort_by_year:
            results = sorted(results, key=lambda x: x.year or '0')

        # Build context from results
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {r.title} ({r.year})]\n{r.text}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Create prompt
        system_prompt = """You are an expert astronomy research assistant specializing in MeerKAT telescope publications.
Answer questions based on the provided research paper excerpts.
Always cite which sources you use (e.g., "According to Source 1...").
When discussing temporal progression or history, organize your answer chronologically by publication year.
If the answer cannot be found in the sources, say so clearly.
Be precise and scientific in your language."""

        user_prompt = f"""Based on these excerpts from MeerKAT research papers:

{context}

---

Question: {query}

Please provide a comprehensive answer based on the sources above."""

        # Generate response
        response = self.llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        answer = response.choices[0].message.content

        # Format sources
        sources = []
        for r in results:
            sources.append({
                "title": r.title,
                "authors": r.authors,
                "year": r.year,
                "similarity": round(r.similarity, 3),
                "arxiv_url": r.arxiv_url,
                "preview": r.text[:200] + "..."
            })

        return {
            "answer": answer,
            "sources": sources
        }

    def interactive_mode(self):
        """Run interactive query mode."""
        print("\n" + "="*60)
        print("MeerKAT Publications RAG System")
        print("="*60)
        print("Ask questions about MeerKAT telescope research.")
        print("Type 'quit' or 'exit' to stop.\n")

        while True:
            try:
                query = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not query:
                continue
            if query.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            print("\nSearching and generating answer...")
            result = self.generate_answer(query)

            print("\n" + "-"*60)
            print("ANSWER:")
            print("-"*60)
            print(result['answer'])

            print("\n" + "-"*60)
            print("SOURCES:")
            print("-"*60)
            for i, src in enumerate(result['sources'], 1):
                print(f"\n{i}. {src['title']}")
                print(f"   {src['authors']} ({src['year']})")
                print(f"   Relevance: {src['similarity']:.1%}")
                if src['arxiv_url']:
                    print(f"   {src['arxiv_url']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MeerKAT Publications RAG System")
    parser.add_argument("query", nargs="?", help="Question to ask (optional)")
    parser.add_argument("--index", action="store_true", help="Re-index papers")
    parser.add_argument("--top-k", type=int, default=5, help="Number of sources to retrieve")

    args = parser.parse_args()

    # Initialize RAG system
    rag = MeerKATRAG()

    # Index if requested or if collection is empty
    if args.index or rag.collection.count() == 0:
        if not PAPERS_FILE.exists():
            print(f"Error: Papers file not found: {PAPERS_FILE}")
            print("Run 'python download_meerkat_papers.py' first.")
            return 1
        rag.index_papers()

    # Handle query
    if args.query:
        result = rag.generate_answer(args.query, top_k=args.top_k)
        print("\nANSWER:")
        print(result['answer'])
        print("\nSOURCES:")
        for i, src in enumerate(result['sources'], 1):
            print(f"{i}. {src['title']} ({src['year']})")
    else:
        rag.interactive_mode()

    return 0


if __name__ == "__main__":
    sys.exit(main())
