#!/usr/bin/env python3
"""
MeerKAT Papers Downloader
Downloads paper metadata from NASA ADS library and prepares for RAG indexing.

Usage:
    1. Copy .env.example to .env and add your ADS_API_TOKEN
    2. Run: python download_meerkat_papers.py
"""

import os
import re
import json
import time
import requests
import pymupdf
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Strip surrounding quotes from value
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value

load_env()


@dataclass
class Paper:
    """Represents a MeerKAT paper with metadata."""
    bibcode: str
    title: str
    abstract: str
    authors: List[str]
    year: int
    publication: str
    doi: Optional[str] = None
    keywords: Optional[List[str]] = None
    arxiv_id: Optional[str] = None
    pdf_path: Optional[str] = None
    full_text: Optional[str] = None

    @property
    def ads_url(self) -> str:
        return f"https://ui.adsabs.harvard.edu/abs/{self.bibcode}/abstract"

    @property
    def arxiv_url(self) -> Optional[str]:
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        return None

    @property
    def arxiv_pdf_url(self) -> Optional[str]:
        if self.arxiv_id:
            return f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"
        return None

    @property
    def author_string(self) -> str:
        if len(self.authors) > 3:
            return ", ".join(self.authors[:3]) + " et al."
        return ", ".join(self.authors)

    def to_document(self) -> str:
        """Convert to text document for embedding."""
        return f"Title: {self.title}\n\nAuthors: {self.author_string}\n\nAbstract: {self.abstract}"


class ADSClient:
    """Client for NASA ADS API."""

    BASE_URL = "https://api.adsabs.harvard.edu/v1"

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv("ADS_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "ADS API token required. Set ADS_API_TOKEN environment variable "
                "or pass api_token parameter.\n"
                "Get your token from: https://ui.adsabs.harvard.edu/user/settings/token"
            )
        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    def get_library_bibcodes(self, library_id: str) -> List[str]:
        """Get all bibcodes from an ADS library."""
        url = f"{self.BASE_URL}/biblib/libraries/{library_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        data = response.json()
        bibcodes = data.get('documents', [])
        print(f"Found {len(bibcodes)} papers in library")
        return bibcodes

    def get_papers_metadata(self, bibcodes: List[str], batch_size: int = 50) -> List[Paper]:
        """Fetch full metadata for papers by bibcode."""
        papers = []

        # Process in batches to avoid API limits
        for i in range(0, len(bibcodes), batch_size):
            batch = bibcodes[i:i + batch_size]
            batch_papers = self._fetch_batch(batch)
            papers.extend(batch_papers)
            print(f"  Fetched {len(papers)}/{len(bibcodes)} papers...")

        return papers

    def _fetch_batch(self, bibcodes: List[str]) -> List[Paper]:
        """Fetch a batch of papers."""
        url = f"{self.BASE_URL}/search/query"

        # Build query for multiple bibcodes
        bibcode_query = " OR ".join(f'bibcode:"{b}"' for b in bibcodes)

        params = {
            'q': bibcode_query,
            'fl': 'bibcode,title,abstract,author,year,pub,doi,keyword,identifier',
            'rows': len(bibcodes)
        }

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        docs = response.json()['response']['docs']
        papers = []

        for doc in docs:
            # Skip papers without abstracts
            if not doc.get('abstract'):
                continue

            # Extract arXiv ID from identifiers
            arxiv_id = None
            identifiers = doc.get('identifier', [])
            for ident in identifiers:
                # Match arXiv IDs like "arXiv:2301.12345" or "2301.12345"
                if 'arXiv:' in ident:
                    arxiv_id = ident.replace('arXiv:', '')
                    break
                elif re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', ident):
                    arxiv_id = ident
                    break

            paper = Paper(
                bibcode=doc.get('bibcode', ''),
                title=doc.get('title', ['Untitled'])[0] if doc.get('title') else 'Untitled',
                abstract=doc.get('abstract', ''),
                authors=doc.get('author', []),
                year=doc.get('year', 0),
                publication=doc.get('pub', 'Unknown'),
                doi=doc.get('doi', [None])[0] if doc.get('doi') else None,
                keywords=doc.get('keyword', []),
                arxiv_id=arxiv_id
            )
            papers.append(paper)

        return papers

    def download_library(self, library_id: str) -> List[Paper]:
        """Download all papers from an ADS library."""
        print(f"Fetching library: {library_id}")
        bibcodes = self.get_library_bibcodes(library_id)

        print(f"Downloading metadata for {len(bibcodes)} papers...")
        papers = self.get_papers_metadata(bibcodes)

        print(f"Successfully retrieved {len(papers)} papers with abstracts")
        return papers


def download_pdfs(papers: List[Paper], output_dir: str = "papers") -> List[Paper]:
    """Download PDFs from arXiv for papers that have arXiv IDs."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    for i, paper in enumerate(papers):
        if not paper.arxiv_id:
            skipped += 1
            continue

        # Create safe filename from arXiv ID
        safe_id = paper.arxiv_id.replace('/', '_')
        pdf_filename = f"{safe_id}.pdf"
        pdf_path = output_path / pdf_filename

        # Skip if already downloaded
        if pdf_path.exists():
            paper.pdf_path = str(pdf_path)
            print(f"  [{i+1}/{len(papers)}] Already exists: {pdf_filename}")
            downloaded += 1
            continue

        # Download from arXiv
        pdf_url = paper.arxiv_pdf_url
        print(f"  [{i+1}/{len(papers)}] Downloading: {pdf_filename}...")

        try:
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()

            # Verify it's a PDF
            if response.headers.get('content-type', '').startswith('application/pdf'):
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                paper.pdf_path = str(pdf_path)
                downloaded += 1
                print(f"           Saved ({len(response.content) // 1024} KB)")
            else:
                print(f"           Not a PDF, skipping")
                failed += 1

            # Be nice to arXiv - wait between requests
            time.sleep(1)

        except requests.RequestException as e:
            print(f"           Failed: {e}")
            failed += 1

    print(f"\nPDF Download Summary:")
    print(f"  Downloaded: {downloaded}")
    print(f"  No arXiv ID: {skipped}")
    print(f"  Failed: {failed}")

    return papers


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = pymupdf.open(pdf_path)
        text_parts = []

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                text_parts.append(text)

        doc.close()

        if text_parts:
            return "\n\n".join(text_parts)
        return None

    except Exception as e:
        print(f"    Error extracting text: {e}")
        return None


def extract_texts(papers: List[Paper], save_txt: bool = True) -> List[Paper]:
    """Extract text from PDFs for all papers that have them."""
    extracted = 0
    skipped = 0
    failed = 0

    for i, paper in enumerate(papers):
        if not paper.pdf_path or not Path(paper.pdf_path).exists():
            skipped += 1
            continue

        print(f"  [{i+1}/{len(papers)}] Extracting: {Path(paper.pdf_path).name}...")

        text = extract_text_from_pdf(paper.pdf_path)

        if text:
            paper.full_text = text
            extracted += 1

            # Optionally save as .txt file alongside PDF
            if save_txt:
                txt_path = Path(paper.pdf_path).with_suffix('.txt')
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)

            word_count = len(text.split())
            print(f"             Extracted {word_count:,} words")
        else:
            failed += 1
            print(f"             No text extracted")

    print(f"\nText Extraction Summary:")
    print(f"  Extracted: {extracted}")
    print(f"  No PDF: {skipped}")
    print(f"  Failed: {failed}")

    return papers


def save_papers(papers: List[Paper], output_path: str = "meerkat_papers.json"):
    """Save papers to JSON file."""
    data = {
        "downloaded_at": datetime.now().isoformat(),
        "count": len(papers),
        "papers": [asdict(p) for p in papers]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(papers)} papers to {output_path}")


def load_papers(input_path: str = "meerkat_papers.json") -> List[Paper]:
    """Load papers from JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    papers = [Paper(**p) for p in data['papers']]
    print(f"Loaded {len(papers)} papers from {input_path}")
    return papers


def main():
    # MeerKAT publications library
    LIBRARY_ID = "wmc9yO6IQ3mUZCPx7MQRxg"
    OUTPUT_FILE = "meerkat_papers.json"
    PDF_DIR = "papers"

    try:
        client = ADSClient()
        papers = client.download_library(LIBRARY_ID)

        # Check for arXiv IDs
        with_arxiv = sum(1 for p in papers if p.arxiv_id)
        print(f"\nFound {with_arxiv}/{len(papers)} papers with arXiv IDs")

        # Download PDFs
        if with_arxiv > 0:
            print(f"\nDownloading PDFs to '{PDF_DIR}/'...")
            papers = download_pdfs(papers, PDF_DIR)

        # Extract text from PDFs
        papers_with_pdfs = sum(1 for p in papers if p.pdf_path)
        if papers_with_pdfs > 0:
            print(f"\nExtracting text from PDFs...")
            papers = extract_texts(papers, save_txt=True)

        # Save metadata (including full text)
        save_papers(papers, OUTPUT_FILE)

        # Print summary
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print("="*60)
        print(f"Total papers: {len(papers)}")
        print(f"Papers with PDFs: {sum(1 for p in papers if p.pdf_path)}")
        print(f"Papers with text: {sum(1 for p in papers if p.full_text)}")

        if papers:
            years = [p.year for p in papers if p.year]
            print(f"Year range: {min(years)} - {max(years)}")

            print("\nSample papers:")
            for paper in papers[:3]:
                print(f"  - {paper.title[:60]}... ({paper.year})")
                print(f"    arXiv: {paper.arxiv_id or 'N/A'}")

        print(f"\nMetadata saved to: {OUTPUT_FILE}")
        print(f"PDFs saved to: {PDF_DIR}/")
        print("Ready for RAG indexing!")

    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except requests.RequestException as e:
        print(f"API error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
