#!/usr/bin/env python3
"""
Simple paper search interface for agentic use.
Returns JSON results for easy parsing.

Usage:
    python search_papers.py "query" [--top-k N] [--no-expand]
"""

import sys
import json
import argparse
from meerkat_rag import MeerKATRAG

def main():
    parser = argparse.ArgumentParser(description="Search MeerKAT papers")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=15, help="Number of results (default: 15 for agentic use)")
    parser.add_argument("--no-expand", action="store_true", help="Disable query expansion")
    parser.add_argument("--full-text", action="store_true", help="Include full chunk text")

    args = parser.parse_args()

    # Suppress initialization messages
    import io
    from contextlib import redirect_stdout

    with redirect_stdout(io.StringIO()):
        rag = MeerKATRAG()

    results = rag.search(args.query, top_k=args.top_k, expand=not args.no_expand)

    output = []
    for r in results:
        item = {
            "title": r.title,
            "authors": r.authors,
            "year": r.year,
            "similarity": round(r.similarity, 3),
            "arxiv_url": r.arxiv_url,
        }
        if args.full_text:
            item["text"] = r.text
        else:
            item["preview"] = r.text[:300] + "..." if len(r.text) > 300 else r.text
        output.append(item)

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
