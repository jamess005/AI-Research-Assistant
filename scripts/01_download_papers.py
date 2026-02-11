#!/usr/bin/env python3
"""
Script 01: Download Papers from ArXiv
Downloads LaTeX source (.tar.gz) files
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from api.arxiv_client import ArxivClient


def main():
    """Download papers - LaTeX source only."""
    print("=" * 80)
    print("ArXiv Paper Downloader - Phase 1")
    print("=" * 80)
    print("\nDownload Strategy:")
    print("  LaTeX source (.tar.gz) only")
    print("  Papers without LaTeX source are skipped")
    
    # Initialize client
    client = ArxivClient(
        config_path=str(project_root / "config" / "arxiv_config.yaml")
    )
    
    # Get number of papers
    print("\nPhase 1 Goal: 100 papers")
    print("  cs.AI and cs.LG categories")
    print("  Published 2020-2025")
    
    try:
        max_results = int(input("\nNumber of papers to download (default 100): ").strip() or "100")
    except ValueError:
        print("Invalid input. Using default of 100 papers.")
        max_results = 100
    
    # Search for papers
    print(f"\nSearching for {max_results} papers...")
    papers = client.search_papers(max_results=max_results)
    
    if not papers:
        print("No papers found. Check configuration.")
        return
    
    # Display sample papers
    print(f"\nFound {len(papers)} papers. Sample:")
    for i, paper in enumerate(papers[:3], 1):
        print(f"\n{i}. {paper['title'][:70]}...")
        print(f"   Authors: {', '.join(paper['authors'][:2])}" +
              (" et al." if len(paper['authors']) > 2 else ""))
        print(f"   Categories: {', '.join(paper['categories'])}")
        print(f"   ArXiv ID: {paper['arxiv_id']}")
    
    # Confirm download
    print(f"\n{'='*80}")
    print(f"Ready to download {len(papers)} papers")
    print(f"{'='*80}")
    confirm = input("Proceed? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Download cancelled.")
        return
    
    # Download papers
    stats = client.download_papers(papers)
    
    print("\n" + "=" * 80)
    print("Download Complete")
    print("=" * 80)
    print(f"\nLaTeX papers downloaded: {stats['latex']}")
    if stats['failed'] > 0:
        print(f"Failed (no LaTeX source): {stats['failed']}")
    print(f"\nEstimated processing time: ~{stats['latex'] * 3 / 60:.1f} minutes")
    
    print("\n\nNext steps:")
    print("1. Run script 02_process_papers.py to parse LaTeX into database")
    print("2. Run script 03_build_index.py to build vector index")


if __name__ == "__main__":
    main()