#!/usr/bin/env python3
"""
05_orchestrator.py - Autonomous RAG Pipeline Orchestrator

Runs the complete pipeline autonomously:
1. Download papers from ArXiv
2. Process papers (LaTeX → Database)
3. Build/rebuild vector index from database

Pipeline stores all data in SQLite database (no JSON intermediates).
"""

import os
import sys
from pathlib import Path
import json
from tqdm import tqdm
import time
import tarfile
import re
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.api.arxiv_client import ArxivClient
from src.data.latex_parser import LatexParser
from src.data.database_schema import PapersDatabase
from src.data.database_chunker import DatabaseChunker
from src.data.embedder import Embedder
from src.data.retrieval.vector_store import VectorStore
from src.data.section_splitter import SectionSplitter
from typing import Optional


class PipelineOrchestrator:
    """Autonomous pipeline orchestrator using database storage."""

    def __init__(self, project_root: Path, config_path: Optional[str] = None):
        self.project_root = project_root
        self.raw_dir = project_root / "data" / "raw"
        self.vector_store_dir = project_root / "data" / "vector_store"
        self.db_path = project_root / "data" / "papers.db"
        self.metadata_file = self.raw_dir / "arxiv_metadata.json"
        self.manifest_file = project_root / "data" / "paper_manifest.json"
        self.failed_papers_file = self.raw_dir / ".failed_papers"

        if config_path:
            self.config_path = str(project_root / "config" / config_path)
        else:
            self.config_path = str(project_root / "config" / "arxiv_config.yaml")

        # Ensure directories exist
        for dir_path in [self.raw_dir, self.vector_store_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.failed_papers: set = self._load_failed_papers()

    def _load_failed_papers(self) -> set:
        """Load the set of known-failed arxiv IDs."""
        if self.failed_papers_file.exists():
            with open(self.failed_papers_file, 'r') as f:
                return {line.strip() for line in f if line.strip()}
        return set()

    def _save_failed_papers(self):
        """Save the failed papers set to disk."""
        with open(self.failed_papers_file, 'w') as f:
            for arxiv_id in sorted(self.failed_papers):
                f.write(f"{arxiv_id}\n")
        print(f"  Tracked {len(self.failed_papers)} failed papers")

    def _update_manifest(self, new_papers: list, failed_papers: Optional[list] = None):
        """Add newly processed/failed papers to the manifest file."""
        from datetime import datetime
        
        # Load existing manifest or create new
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {
                'description': 'Complete paper manifest - all papers downloaded and processed',
                'last_updated': '',
                'total_papers': 0,
                'processed_papers': 0,
                'unprocessed_papers': 0,
                'papers': []
            }
        
        # Build set of existing arxiv_ids for dedup
        existing_ids = {p['arxiv_id'] for p in manifest['papers']}
        
        added = 0
        for paper in new_papers:
            if paper['arxiv_id'] not in existing_ids:
                manifest['papers'].append(paper)
                existing_ids.add(paper['arxiv_id'])
                added += 1
        
        # Add failed papers
        if failed_papers:
            for paper in failed_papers:
                if paper['arxiv_id'] not in existing_ids:
                    manifest['papers'].append(paper)
                    existing_ids.add(paper['arxiv_id'])
                    added += 1
        
        # Update counts
        manifest['total_papers'] = len(manifest['papers'])
        manifest['processed_papers'] = sum(1 for p in manifest['papers'] if p['status'] == 'processed')
        manifest['unprocessed_papers'] = sum(1 for p in manifest['papers'] if p['status'] != 'processed')
        manifest['last_updated'] = datetime.now().isoformat()
        
        with open(self.manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        if added > 0:
            print(f"  Manifest updated: +{added} papers (total: {manifest['total_papers']})")

    def validate_and_cleanup_tarballs(self) -> tuple:
        """Validate tar.gz files and remove corrupted ones."""
        latex_files = list(self.raw_dir.glob("*.tar.gz"))
        valid_files = []
        removed_count = 0

        for tar_file in latex_files:
            try:
                with tarfile.open(tar_file, 'r:gz') as tar:
                    tar.getmembers()
                valid_files.append(tar_file)
            except (tarfile.ReadError, EOFError, OSError, Exception) as e:
                arxiv_id = tar_file.stem.replace('.tar', '')
                print(f"  Removing corrupted: {tar_file.name} ({type(e).__name__})")
                removed_count += 1
                tar_file.unlink(missing_ok=True)
                # Track as failed so we don't try to download again
                self.failed_papers.add(arxiv_id)

        return valid_files, removed_count

    def _get_existing_arxiv_ids(self) -> set:
        """Collect all arxiv_ids we already have (in DB, downloading, or failed)."""
        ids = set()
        # Papers still being downloaded (tar.gz not yet processed)
        for f in self.raw_dir.glob("*.tar.gz"):
            ids.add(f.stem.replace('.tar', ''))
        # Papers already in database
        if self.db_path.exists():
            with PapersDatabase(str(self.db_path)) as db:
                for paper in db.get_all_papers():
                    ids.add(paper['arxiv_id'])
        # Known failed papers
        ids.update(self.failed_papers)
        return ids

    def step_1_download_papers(self, num_papers: int) -> int:
        """Download NEW papers from ArXiv."""
        print("=" * 80)
        print("STEP 1: Download Papers from ArXiv")
        print("=" * 80)

        existing_ids = self._get_existing_arxiv_ids()

        print(f"\nTarget: {num_papers} NEW papers")
        print(f"Already have: {len(existing_ids)} papers (downloaded + failed)")

        client = ArxivClient(config_path=self.config_path)

        config_name = Path(self.config_path).name
        if config_name != "arxiv_config.yaml":
            print(f"Using config: {config_name}")

        new_papers = []
        offset = 0
        batch_size = min(100, num_papers)
        max_offset = len(existing_ids) + num_papers * 10
        consecutive_empty = 0

        print(f"\nSearching for papers (max offset: {max_offset})...")
        while len(new_papers) < num_papers and offset < max_offset:
            if offset > 0:
                print(f"  Waiting 3 seconds (rate limit)...")
                time.sleep(3)

            papers = client.search_papers(max_results=batch_size, start=offset)

            if not papers:
                print(f"  No more papers available at offset {offset}")
                break

            batch_new = 0
            for p in papers:
                if p['arxiv_id'] not in existing_ids:
                    new_papers.append(p)
                    existing_ids.add(p['arxiv_id'])
                    batch_new += 1

            print(f"  Searched offset {offset}-{offset+len(papers)}: "
                  f"{len(new_papers)}/{num_papers} new papers found")

            if len(new_papers) >= num_papers:
                break

            # Exponential skip: if consecutive batches find nothing, jump ahead
            if batch_new == 0:
                consecutive_empty += 1
                skip = batch_size * (2 ** consecutive_empty)
                print(f"  No new papers in batch, skipping ahead {skip}...")
                offset += skip
            else:
                consecutive_empty = 0
                offset += batch_size

        new_papers = new_papers[:num_papers]

        if not new_papers:
            print("No new papers found.")
            return 0

        print(f"Found {len(new_papers)} new papers to download")

        print("\nDownloading papers...")
        stats = client.download_papers(new_papers)

        total_downloaded = stats['latex'] + stats['pdf']
        print(f"\nDownloaded {total_downloaded} new papers")
        print(f"  LaTeX sources: {stats['latex']}")
        if stats['failed'] > 0:
            print(f"  Failed: {stats['failed']}")

        return total_downloaded

    @staticmethod
    def _count_sentences(text: str) -> int:
        """Count sentences in text."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return len([s for s in sentences if s.strip()])

    def step_2_process_papers(self, use_splitter: bool = True) -> int:
        """Process papers: LaTeX → Database."""
        print("\n" + "=" * 80)
        print("STEP 2: Process Papers (LaTeX → Database)")
        print("=" * 80)

        # Validate and cleanup corrupted files
        print("\nValidating tar.gz files...")
        latex_files, removed = self.validate_and_cleanup_tarballs()

        if removed > 0:
            print(f"Removed {removed} corrupted tar.gz files")

        print(f"Valid LaTeX sources: {len(latex_files)}")

        if not latex_files:
            print("No files to process!")
            return 0

        # Load metadata
        metadata_dict = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                if isinstance(metadata, dict):
                    metadata_dict = metadata
                elif isinstance(metadata, list):
                    for item in metadata:
                        arxiv_id = item.get('arxiv_id', '')
                        metadata_dict[arxiv_id] = item

        # Find NEW papers to process (not in DB and not failed)
        latex_to_process = []

        with PapersDatabase(str(self.db_path)) as db:
            db.create_tables()
            for latex_file in latex_files:
                arxiv_id = latex_file.stem.replace('.tar', '')
                if arxiv_id in self.failed_papers:
                    continue
                if not db.get_paper(arxiv_id):
                    latex_to_process.append((latex_file, arxiv_id))

        if not latex_to_process:
            print("\nAll papers already processed!")
            return 0

        print(f"\nNew papers to process: {len(latex_to_process)}")
        if self.failed_papers:
            print(f"  Skipping {len(self.failed_papers)} known-failed papers")

        # Optionally load section splitter
        splitter = None
        if use_splitter:
            try:
                splitter = SectionSplitter(
                    model_name="meta-llama/Llama-3.2-3B-Instruct",
                    max_section_words=5000
                )
                splitter.load_model()
                print("  Section splitter loaded")
            except Exception as e:
                print(f"  Warning: Could not load splitter ({e})")
                splitter = None

        successful = 0
        failed_this_run = 0
        processed_papers = []
        failed_papers_list = []

        latex_parser = LatexParser()

        with PapersDatabase(str(self.db_path)) as db:
            db.create_tables()

            for latex_file, arxiv_id in tqdm(latex_to_process, desc="Processing"):
                try:
                    paper_data = latex_parser.parse_paper(
                        str(latex_file),
                        arxiv_id,
                        metadata_dict.get(arxiv_id)
                    )

                    if paper_data:
                        # Insert paper metadata
                        db.insert_paper(paper_data)

                        # Insert sections
                        for section_name, section_text in paper_data.get('sections', {}).items():
                            if not section_text or not section_text.strip():
                                continue

                            word_count = len(section_text.split())
                            sentence_count = self._count_sentences(section_text)

                            if word_count > 5000 and splitter:
                                subsections = splitter.split_section_intelligent(
                                    section_text=section_text,
                                    section_name=section_name,
                                    paper_title=paper_data.get('title', '')
                                )
                                for idx, (sub_text, description) in enumerate(subsections):
                                    db.insert_section({
                                        'arxiv_id': arxiv_id,
                                        'section_name': f"{section_name}_part_{idx}",
                                        'section_text': sub_text,
                                        'subsection_index': idx,
                                        'subsection_description': description,
                                        'word_count': len(sub_text.split()),
                                        'sentence_count': self._count_sentences(sub_text),
                                        'is_split': True,
                                        'original_section_name': section_name
                                    })
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

                        successful += 1
                        
                        # Track for manifest
                        processed_papers.append({
                            'arxiv_id': arxiv_id,
                            'title': paper_data.get('title', ''),
                            'authors': paper_data.get('authors', ''),
                            'categories': paper_data.get('categories', ''),
                            'added_to_db': None,  # Will use current timestamp
                            'status': 'processed'
                        })
                        
                        # Delete tar.gz after successful processing
                        latex_file.unlink(missing_ok=True)
                        
                    else:
                        self.failed_papers.add(arxiv_id)
                        failed_this_run += 1
                        # Track for manifest
                        failed_papers_list.append({
                            'arxiv_id': arxiv_id,
                            'title': None,
                            'authors': None,
                            'categories': None,
                            'added_to_db': None,
                            'status': 'failed_processing'
                        })
                        # Delete tar.gz for failed papers too (tracked in .failed_papers)
                        latex_file.unlink(missing_ok=True)
                except Exception as e:
                    print(f"\n  Error processing {arxiv_id}: {e}")
                    self.failed_papers.add(arxiv_id)
                    failed_this_run += 1
                    # Track for manifest
                    failed_papers_list.append({
                        'arxiv_id': arxiv_id,
                        'title': None,
                        'authors': None,
                        'categories': None,
                        'added_to_db': None,
                        'status': 'failed_processing'
                    })
                    # Delete tar.gz for errored papers too
                    latex_file.unlink(missing_ok=True)

        # Unload splitter
        if splitter:
            splitter.unload_model()
        
        # Update manifest with newly processed AND failed papers
        if processed_papers or failed_papers_list:
            self._update_manifest(processed_papers, failed_papers_list)

        print(f"\nProcessed {successful} new papers")
        if failed_this_run > 0:
            print(f"  Failed this run: {failed_this_run}")
        if latex_parser.fallback_count > 0:
            print(f"  LaTeX fallback used: {latex_parser.fallback_count}")

        return successful

    def step_3_build_index(self, batch_size: int = 128) -> int:
        """Build vector index from database (incremental - only new papers)."""
        print("\n" + "=" * 80)
        print("STEP 3: Build Vector Index from Database")
        print("=" * 80)

        if not self.db_path.exists():
            print("Database not found!")
            return 0

        # Check which papers are already indexed
        store = VectorStore(
            persist_directory=str(self.vector_store_dir),
            collection_name="research_papers"
        )
        current_count = store.count()
        print(f"\nCurrent index: {current_count:,} chunks")

        indexed_arxiv_ids = store.get_existing_arxiv_ids()
        print(f"Papers in index: {len(indexed_arxiv_ids)}")

        # Find papers in DB that aren't in the index
        with PapersDatabase(str(self.db_path)) as db:
            all_papers = db.get_all_papers()
            db_arxiv_ids = {p['arxiv_id'] for p in all_papers}

        new_arxiv_ids = db_arxiv_ids - indexed_arxiv_ids
        
        if not new_arxiv_ids:
            print(f"Vector index is up to date ({current_count:,} chunks)")
            return 0

        print(f"New papers to index: {len(new_arxiv_ids)}")

        # Generate chunks ONLY for new papers
        print("\nChunking new papers...")
        with DatabaseChunker(
            db_path=str(self.db_path),
            sentences_per_chunk=3,
            overlap_sentences=0
        ) as chunker:
            new_chunks = chunker.chunk_papers(sorted(new_arxiv_ids), show_progress=True)

        print(f"Generated {len(new_chunks)} new chunks")

        if not new_chunks:
            print("No new chunks generated!")
            return 0

        # Load embedder
        print("\nLoading embedding model (GPU)...")
        embedder = Embedder(
            model_name=os.environ.get('EMBEDDING_MODEL_NAME', 'all-mpnet-base-v2'),
            cache_dir=str(self.project_root / os.environ.get('EMBEDDING_CACHE_DIR', 'models/embedding')),
            force_gpu=True
        )

        # Generate embeddings ONLY for new chunks
        chunk_texts = [chunk['chunk_text'] for chunk in new_chunks]
        print(f"Embedding {len(chunk_texts)} new chunks...")
        embeddings = embedder.embed_text(
            chunk_texts,
            batch_size=batch_size,
            show_progress=True
        )

        print(f"Generated {len(embeddings)} embeddings ({embeddings.shape[1]}-dim)")

        # Add new chunks to existing index (no reset!)
        chunk_ids = [chunk['chunk_id'] for chunk in new_chunks]
        documents = [chunk['parent_text'] for chunk in new_chunks]

        metadatas = []
        for chunk in new_chunks:
            metadata = {
                'chunk_id': chunk['chunk_id'],
                'chunk_text': chunk['chunk_text'],
                'chunk_word_count': chunk['chunk_word_count'],
                'chunk_sentence_count': chunk['chunk_sentence_count'],
                'parent_section_name': chunk['parent_section_name'],
                'parent_word_count': chunk['parent_word_count'],
                'parent_sentence_count': chunk['parent_sentence_count'],
                'db_section_id': chunk['db_section_id'],
                'is_split': chunk['is_split'],
                'subsection_index': chunk['subsection_index'],
                'subsection_description': chunk['subsection_description'],
                'original_section_name': chunk['original_section_name'],
                'arxiv_id': chunk['arxiv_id'],
                'title': chunk['title'],
                'authors': ', '.join(chunk['authors']) if isinstance(chunk['authors'], list) else chunk['authors'],
                'categories': ', '.join(chunk['categories']) if isinstance(chunk['categories'], list) else chunk['categories'],
                'parsing_method': chunk['parsing_method']
            }
            metadatas.append(metadata)

        # Add in batches (incremental - no reset_collection!)
        batch_size_store = 100
        print(f"Adding {len(new_chunks)} chunks to index...")
        for i in tqdm(range(0, len(new_chunks), batch_size_store), desc="Indexing"):
            end_idx = min(i + batch_size_store, len(new_chunks))
            store.add_documents(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=chunk_ids[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist()
            )

        final_count = store.count()
        print(f"Index updated: {current_count:,} → {final_count:,} chunks (+{final_count - current_count:,})")
        return len(new_chunks)


def main():
    """Run autonomous pipeline."""
    print("=" * 80)
    print("AUTONOMOUS RAG PIPELINE ORCHESTRATOR")
    print("=" * 80)

    print("\nPipeline steps:")
    print("  1. Download papers from ArXiv")
    print("  2. Process papers (LaTeX → Database)")
    print("  3. Build vector index from database")

    print("\nINCREMENTAL MODE:")
    print("  - Adds NEW papers on top of existing data")
    print("  - Skips already processed papers")
    print("  - Rebuilds vector store only when needed")

    project_root = Path(__file__).parent.parent

    # Choose download mode
    print("\n" + "=" * 80)
    print("DOWNLOAD MODE")
    print("=" * 80)
    print("\n1. General papers (cs.AI and cs.LG)")
    print("2. Targeted: Optimization & Training")
    print("3. Targeted: Hyperparameter Tuning")
    print("4. Targeted: Model Compression")
    print("5. Targeted: Transfer Learning")
    print("6. Targeted: Interpretability")
    print("7. Targeted: Robustness & Security")

    mode = input("\nChoose mode (1-7, default 1): ").strip() or "1"

    config_map = {
        "1": (None, "General papers"),
        "2": ("arxiv_config_targeted.yaml", "Optimization & Training"),
        "3": ("arxiv_config_targeted.yaml", "Hyperparameter Tuning"),
        "4": ("arxiv_config_targeted.yaml", "Model Compression"),
        "5": ("arxiv_config_targeted.yaml", "Transfer Learning"),
        "6": ("arxiv_config_targeted.yaml", "Interpretability"),
        "7": ("arxiv_config_targeted.yaml", "Robustness & Security")
    }

    if mode not in config_map:
        print("Invalid mode. Using general papers.")
        mode = "1"

    config_file, mode_name = config_map[mode]

    default_count = 300 if mode != "1" else 100
    try:
        num_papers = input(f"\nHow many papers to download? (default {default_count}): ").strip()
        num_papers = int(num_papers) if num_papers else default_count
    except ValueError:
        num_papers = default_count

    # If targeted mode, create temp config
    if mode != "1":
        query_map = {
            "2": "learning rate OR hyperparameter optimization OR gradient descent OR optimizer OR training dynamics OR learning rate schedule",
            "3": "hyperparameter tuning OR hyperparameter optimization OR neural architecture search OR automl OR bayesian optimization",
            "4": "model compression OR quantization OR pruning OR knowledge distillation OR efficient inference",
            "5": "transfer learning OR domain adaptation OR fine-tuning OR pre-training OR few-shot learning OR meta-learning",
            "6": "interpretability OR explainability OR attention visualization OR saliency maps OR feature attribution",
            "7": "adversarial robustness OR adversarial training OR adversarial examples OR model security OR certified robustness"
        }

        base_config_path = project_root / "config" / "arxiv_config_targeted.yaml"
        with open(base_config_path) as f:
            config = yaml.safe_load(f)

        config['search']['query'] = query_map[mode]
        config['search']['max_results'] = num_papers

        temp_config_path = project_root / "config" / f"_temp_mode{mode}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        config_file = f"_temp_mode{mode}.yaml"

    print(f"\n" + "=" * 80)
    print(f"Mode: {mode_name}")
    print(f"Papers: {num_papers}")
    print("=" * 80)

    confirm = input("\nProceed with pipeline? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Pipeline cancelled.")
        return

    orchestrator = PipelineOrchestrator(project_root, config_path=config_file)

    start_time = time.time()

    try:
        downloaded = orchestrator.step_1_download_papers(num_papers)
        processed = orchestrator.step_2_process_papers()
        indexed = orchestrator.step_3_build_index()

        orchestrator._save_failed_papers()

        elapsed = time.time() - start_time

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)

        print(f"\nThis run:")
        print(f"  Papers downloaded: {downloaded}")
        print(f"  Papers processed: {processed}")
        print(f"  Chunks indexed: {indexed}")
        print(f"  Time elapsed: {elapsed/60:.1f} minutes")

        # Database totals
        with PapersDatabase(str(orchestrator.db_path)) as db:
            print(f"\nDatabase totals:")
            print(f"  Papers: {db.count_papers()}")
            print(f"  Sections: {db.count_sections()}")

        # Vector store totals
        store = VectorStore(
            persist_directory=str(orchestrator.vector_store_dir),
            collection_name="research_papers"
        )
        print(f"  Vector store: {store.count():,} chunks")

        print(f"  Failed (tracked): {len(orchestrator.failed_papers)}")

        print("\n\nNext steps:")
        print("  python scripts/04_query.py --interactive")
        print("\nRun again to fetch the next batch of papers!")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        orchestrator._save_failed_papers()
    except Exception as e:
        print(f"\n\nPipeline error: {e}")
        import traceback
        traceback.print_exc()
        orchestrator._save_failed_papers()
    finally:
        if mode != "1":
            temp_config_path = project_root / "config" / f"_temp_mode{mode}.yaml"
            if temp_config_path.exists():
                temp_config_path.unlink()


if __name__ == "__main__":
    main()