#!/usr/bin/env python3
"""
Script 02: Process Papers
Parse LaTeX source and save directly to SQLite database.

Pipeline:
  tar.gz → LaTeX parser → database (papers + sections tables)
  Large sections (>5000 words) are intelligently split into subsections.
"""

import sys
from pathlib import Path
import json
import tarfile
import re
import argparse
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.latex_parser import LatexParser
from src.data.database_schema import PapersDatabase
from src.data.section_splitter import SectionSplitter


def validate_and_cleanup_tarballs(raw_dir: Path) -> tuple:
    """
    Validate tar.gz files and remove corrupted ones.
    Returns (valid_files, removed_count)
    """
    latex_files = list(raw_dir.glob("*.tar.gz"))
    valid_files = []
    removed_count = 0

    for tar_file in latex_files:
        try:
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.getmembers()
            valid_files.append(tar_file)
        except (tarfile.ReadError, EOFError, OSError, Exception) as e:
            print(f"  Removing corrupted file: {tar_file.name} ({type(e).__name__})")
            tar_file.unlink()
            removed_count += 1

    return valid_files, removed_count


def count_sentences(text: str) -> int:
    """Count sentences in text."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([s for s in sentences if s.strip()])


def save_paper_to_database(
    db: PapersDatabase,
    paper_data: dict,
    splitter: Optional[SectionSplitter] = None,
    split_threshold: int = 5000
) -> tuple:
    """
    Save a parsed paper directly to the database.

    Args:
        db: Connected PapersDatabase instance
        paper_data: Output from LatexParser.parse_paper()
        splitter: Optional SectionSplitter for large sections
        split_threshold: Word count threshold for splitting

    Returns:
        (sections_added, sections_split) counts
    """
    arxiv_id = paper_data['arxiv_id']

    # Insert paper metadata
    db.insert_paper(paper_data)

    sections_added = 0
    sections_split = 0

    for section_name, section_text in paper_data.get('sections', {}).items():
        if not section_text or not section_text.strip():
            continue

        word_count = len(section_text.split())
        sentence_count = count_sentences(section_text)

        # Check if section needs splitting
        if word_count > split_threshold and splitter:
            subsections = splitter.split_section_intelligent(
                section_text=section_text,
                section_name=section_name,
                paper_title=paper_data.get('title', '')
            )

            for idx, (sub_text, description) in enumerate(subsections):
                sub_word_count = len(sub_text.split())
                sub_sentence_count = count_sentences(sub_text)

                db.insert_section({
                    'arxiv_id': arxiv_id,
                    'section_name': f"{section_name}_part_{idx}",
                    'section_text': sub_text,
                    'subsection_index': idx,
                    'subsection_description': description,
                    'word_count': sub_word_count,
                    'sentence_count': sub_sentence_count,
                    'is_split': True,
                    'original_section_name': section_name
                })
                sections_added += 1

            sections_split += 1
        else:
            db.insert_section({
                'arxiv_id': arxiv_id,
                'section_name': section_name,
                'section_text': section_text,
                'subsection_index': 0,
                'subsection_description': '',
                'word_count': word_count,
                'sentence_count': sentence_count,
                'is_split': False,
                'original_section_name': ''
            })
            sections_added += 1

    return sections_added, sections_split


def main():
    """Process papers with LaTeX pipeline, saving directly to database."""
    parser = argparse.ArgumentParser(
        description="Process LaTeX papers and save to database"
    )
    parser.add_argument(
        "--no-split", action="store_true",
        help="Disable intelligent section splitting"
    )
    parser.add_argument(
        "--split-threshold", type=int, default=5000,
        help="Word count threshold for splitting sections (default: 5000)"
    )
    parser.add_argument(
        "--db-path", type=str, default="data/papers.db",
        help="Path to database file (default: data/papers.db)"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Paper Processing - LaTeX → Database Pipeline")
    print("=" * 80)

    # Set up paths
    raw_dir = project_root / "data" / "raw"
    metadata_file = raw_dir / "arxiv_metadata.json"
    db_path = project_root / args.db_path

    # Validate and cleanup corrupted tar.gz files
    print("\nValidating tar.gz files...")
    latex_files, removed = validate_and_cleanup_tarballs(raw_dir)

    if removed > 0:
        print(f"  Removed {removed} corrupted tar.gz files")

    print(f"  Valid LaTeX sources: {len(latex_files)}")

    if not latex_files:
        print("\nNo files to process. Run 01_download_papers.py first.")
        return

    # Load metadata (handle both dict and list formats)
    metadata_dict = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            if isinstance(metadata, dict):
                metadata_dict = metadata
            elif isinstance(metadata, list):
                for item in metadata:
                    arxiv_id = item.get('arxiv_id', '')
                    metadata_dict[arxiv_id] = item

    # Connect to database and check which papers are already processed
    with PapersDatabase(str(db_path)) as db:
        db.create_tables()

        latex_to_process = []
        for latex_file in latex_files:
            arxiv_id = latex_file.stem.replace('.tar', '')
            existing = db.get_paper(arxiv_id)
            if not existing:
                latex_to_process.append((latex_file, arxiv_id))

    print(f"\n  Already in database: {len(latex_files) - len(latex_to_process)}")
    print(f"  To process: {len(latex_to_process)}")

    if not latex_to_process:
        print("\nAll papers already processed!")
        return

    # Confirm
    if not args.yes:
        confirm = input(f"\nProcess {len(latex_to_process)} papers? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Processing cancelled.")
            return

    # Optionally load section splitter
    splitter = None
    if not args.no_split:
        print("\nLoading section splitter for large sections...")
        try:
            splitter = SectionSplitter(
                model_name="meta-llama/Llama-3.2-3B-Instruct",
                max_section_words=args.split_threshold
            )
            splitter.load_model()
            print("  Section splitter ready")
        except Exception as e:
            print(f"  Warning: Could not load splitter ({e})")
            print("  Large sections will be stored as-is")
            splitter = None

    # Process papers
    print("\n" + "=" * 80)
    print("Processing LaTeX Sources → Database")
    print("=" * 80)

    latex_parser = LatexParser()
    total_sections = 0
    total_split = 0
    successful = 0
    failed = 0

    with PapersDatabase(str(db_path)) as db:
        db.create_tables()

        for i, (latex_file, arxiv_id) in enumerate(latex_to_process, 1):
            print(f"\n[{i}/{len(latex_to_process)}] {latex_file.name}...")

            try:
                paper_data = latex_parser.parse_paper(
                    str(latex_file),
                    arxiv_id,
                    metadata_dict.get(arxiv_id)
                )

                if paper_data:
                    sections_added, sections_split = save_paper_to_database(
                        db=db,
                        paper_data=paper_data,
                        splitter=splitter,
                        split_threshold=args.split_threshold
                    )

                    total_sections += sections_added
                    total_split += sections_split
                    successful += 1
                    split_info = f" ({sections_split} split)" if sections_split else ""
                    print(f"  {paper_data['word_count']} words, "
                          f"{sections_added} sections{split_info}")
                    
                    # Delete tar.gz after successful processing
                    latex_file.unlink(missing_ok=True)
                else:
                    failed += 1
                    print(f"  Failed to parse")
                    latex_file.unlink(missing_ok=True)

            except Exception as e:
                failed += 1
                print(f"  Error: {e}")
                latex_file.unlink(missing_ok=True)

    # Unload splitter
    if splitter:
        splitter.unload_model()

    # Summary
    print("\n" + "=" * 80)
    print("Processing Complete")
    print("=" * 80)
    print(f"  Processed: {successful}/{len(latex_to_process)}")
    if failed > 0:
        print(f"  Failed: {failed}")
    print(f"  Sections added: {total_sections}")
    if total_split > 0:
        print(f"  Sections split: {total_split}")
    if latex_parser.fallback_count > 0:
        print(f"  LaTeX fallback used: {latex_parser.fallback_count}")

    # Show database totals
    with PapersDatabase(str(db_path)) as db:
        print(f"\n  Database totals:")
        print(f"    Papers: {db.count_papers()}")
        print(f"    Sections: {db.count_sections()}")

    print("\n\nNext step: python scripts/03_build_index.py")


if __name__ == "__main__":
    main()