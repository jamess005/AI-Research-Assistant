#!/usr/bin/env python3
"""
Rebuild Vector Index from Database

Builds vector index directly from database sections:
- No intermediate chunk JSON files needed
- Chunks generated on-the-fly from database sections
- Stores section_id references for fast retrieval
- GPU-accelerated embeddings
"""

import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.database_schema import PapersDatabase
from src.data.database_chunker import DatabaseChunker
from src.data.embedder import Embedder
from src.data.retrieval.vector_store import VectorStore


def rebuild_index_from_database(
    db_path: str,
    vector_store_dir: str,
    sentences_per_chunk: int = 3,
    batch_size: int = 128,
    force_rebuild: bool = False
):
    """
    Build vector index from database (incremental by default).
    
    Args:
        db_path: Path to papers database
        vector_store_dir: Directory for vector store
        sentences_per_chunk: Sentences per chunk
        batch_size: Embedding batch size
        force_rebuild: If True, reset and rebuild entire index
    """
    
    print("="*80)
    print("Build Vector Index from Database")
    print("="*80)
    print(f"\nDatabase: {db_path}")
    print(f"Vector store: {vector_store_dir}")
    print(f"Sentences per chunk: {sentences_per_chunk}")
    print(f"Mode: {'FULL REBUILD' if force_rebuild else 'INCREMENTAL'}")
    
    # Check what's already indexed
    store = VectorStore(
        persist_directory=vector_store_dir,
        collection_name="research_papers"
    )
    current_count = store.count()
    print(f"\nCurrent index: {current_count:,} chunks")
    
    if force_rebuild:
        # Full rebuild mode - chunk everything
        print("\n" + "="*80)
        print("Step 1: Loading Database and Chunking ALL Papers")
        print("="*80)
        
        with DatabaseChunker(
            db_path=db_path,
            sentences_per_chunk=sentences_per_chunk,
            overlap_sentences=0
        ) as chunker:
            print("Generating chunks from all database sections...")
            all_chunks = chunker.chunk_all_papers(show_progress=True)
        
        print(f"\nGenerated {len(all_chunks)} chunks")
        store.reset_collection()
        print("Collection reset (clean rebuild)")
        
    else:
        # Incremental mode - only new papers
        indexed_arxiv_ids = store.get_existing_arxiv_ids()
        print(f"Papers already indexed: {len(indexed_arxiv_ids)}")
        
        with PapersDatabase(db_path) as db:
            all_papers = db.get_all_papers()
            db_arxiv_ids = {p['arxiv_id'] for p in all_papers}
        
        new_arxiv_ids = db_arxiv_ids - indexed_arxiv_ids
        
        if not new_arxiv_ids:
            print(f"\nVector index is up to date ({current_count:,} chunks)")
            return
        
        print(f"New papers to index: {len(new_arxiv_ids)}")
        
        print("\n" + "="*80)
        print("Step 1: Chunking New Papers Only")
        print("="*80)
        
        with DatabaseChunker(
            db_path=db_path,
            sentences_per_chunk=sentences_per_chunk,
            overlap_sentences=0
        ) as chunker:
            print(f"Generating chunks for {len(new_arxiv_ids)} new papers...")
            all_chunks = chunker.chunk_papers(sorted(new_arxiv_ids), show_progress=True)
        
        print(f"\nGenerated {len(all_chunks)} new chunks")
        
        # Compute statistics from already-generated chunks (no re-chunking)
        if all_chunks:
            chunk_wc = [c['chunk_word_count'] for c in all_chunks]
            parent_wc = [c['parent_word_count'] for c in all_chunks]
            unique_parents = len(set(c.get('parent_section_id', c['chunk_id']) for c in all_chunks))
            print(f"\nChunk Statistics:")
            print(f"  Total chunks: {len(all_chunks)}")
            print(f"  Unique parent sections: {unique_parents}")
            print(f"  Avg chunk size: {sum(chunk_wc)/len(chunk_wc):.1f} words")
            print(f"  Avg parent size: {sum(parent_wc)/len(parent_wc):.1f} words")
        
    # Pre-flight validation (catch issues BEFORE expensive embedding)
    print("\n" + "="*80)
    print("Validation: Pre-flight Checks")
    print("="*80)
    
    validation_errors = []
    
    # Check required fields
    required_fields = [
        'chunk_id', 'chunk_text', 'chunk_word_count', 'chunk_sentence_count',
        'parent_section_name', 'parent_word_count', 'parent_text',
        'db_section_id', 'is_split', 'subsection_index', 'subsection_description',
        'original_section_name', 'arxiv_id', 'title', 'authors', 'categories',
        'parsing_method'
    ]
    sample = all_chunks[0]
    missing = [f for f in required_fields if f not in sample]
    if missing:
        validation_errors.append(f"Missing fields in chunks: {missing}")
    else:
        print("  All required metadata fields present")
    
    # Check for list/dict values (ChromaDB rejects these)
    for key, val in sample.items():
        if isinstance(val, (list, dict)):
            validation_errors.append(f"Field '{key}' is {type(val).__name__} - ChromaDB requires str/int/float/bool/None")
    if not any("list" in e or "dict" in e for e in validation_errors):
        print("  No list/dict values (ChromaDB compatible)")
    
    # Check for duplicate chunk IDs
    chunk_id_set = set()
    duplicates = 0
    for c in all_chunks:
        cid = c['chunk_id']
        if cid in chunk_id_set:
            duplicates += 1
        chunk_id_set.add(cid)
    if duplicates > 0:
        validation_errors.append(f"Found {duplicates} duplicate chunk IDs")
    else:
        print(f"  All {len(all_chunks)} chunk IDs are unique")
    
    # Check for empty texts
    empty_chunks = sum(1 for c in all_chunks if not c.get('chunk_text', '').strip())
    if empty_chunks > 0:
        validation_errors.append(f"{empty_chunks} chunks have empty text")
    else:
        print("  No empty chunk texts")
    
    if validation_errors:
        print("\n❌ VALIDATION FAILED:")
        for e in validation_errors:
            print(f"  - {e}")
        print("\nAborting to prevent wasting time on embeddings.")
        raise RuntimeError(f"Pre-flight validation failed: {validation_errors[0]}")
    
    print("\nAll pre-flight checks passed!")
    
    # Step 2: Load embedder
    print("\n" + "="*80)
    print("Step 2: Loading Embedding Model (GPU)")
    print("="*80)
    
    embedder = Embedder(
        model_name='all-mpnet-base-v2',
        cache_dir=str(project_root / "models" / "embedding"),
        force_gpu=True
    )
    
    print(f"Model loaded on {embedder.device}")
    
    # Step 3: Generate embeddings
    print("\n" + "="*80)
    print("Step 3: Generating Embeddings (GPU)")
    print("="*80)
    
    # Extract chunk texts
    chunk_texts = [chunk['chunk_text'] for chunk in all_chunks]
    
    print(f"Embedding {len(chunk_texts)} chunks...")
    embeddings = embedder.embed_text(
        chunk_texts,
        batch_size=batch_size,
        show_progress=True
    )
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"  Shape: {embeddings.shape}")
    
    # Step 4: Build vector store
    print("\n" + "="*80)
    print("Step 4: Adding to Vector Store")
    print("="*80)
    
    # Prepare data
    chunk_ids = [chunk['chunk_id'] for chunk in all_chunks]
    
    # Prepare metadata - include chunk info + reference to database section
    metadatas = []
    for chunk in all_chunks:
        metadata = {
            # Chunk info
            'chunk_id': chunk['chunk_id'],
            'chunk_text': chunk['chunk_text'],
            'chunk_word_count': chunk['chunk_word_count'],
            'chunk_sentence_count': chunk['chunk_sentence_count'],
            
            # Parent section info (for immediate display)
            'parent_section_name': chunk['parent_section_name'],
            'parent_word_count': chunk['parent_word_count'],
            'parent_sentence_count': chunk['parent_sentence_count'],
            
            # Database references (for retrieval)
            'db_section_id': chunk['db_section_id'],
            'is_split': chunk['is_split'],
            'subsection_index': chunk['subsection_index'],
            'subsection_description': chunk['subsection_description'],
            'original_section_name': chunk['original_section_name'],
            
            # Paper metadata (convert lists to strings for ChromaDB)
            'arxiv_id': chunk['arxiv_id'],
            'title': chunk['title'],
            'authors': ', '.join(chunk['authors']) if isinstance(chunk['authors'], list) else chunk['authors'],
            'categories': ', '.join(chunk['categories']) if isinstance(chunk['categories'], list) else chunk['categories'],
            'parsing_method': chunk['parsing_method']
        }
        metadatas.append(metadata)
        
    # Documents: Store PARENT TEXT (not chunk text)
    documents = [chunk['parent_text'] for chunk in all_chunks]
    
    # Add to vector store in batches
    batch_size_store = 100
    print(f"Adding {len(all_chunks)} chunks to vector store...")
    
    for i in tqdm(range(0, len(all_chunks), batch_size_store), desc="Indexing"):
        end_idx = min(i + batch_size_store, len(all_chunks))
        
        store.add_documents(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=chunk_ids[i:end_idx],
            embeddings=embeddings[i:end_idx].tolist()
        )
        
    print(f"Vector store built successfully")
    print(f"  Total chunks indexed: {store.count()}")
    
    # Step 5: Test retrieval
    print("\n" + "="*80)
    print("Step 5: Testing Retrieval")
    print("="*80)
    
    test_query = "What are transformers in machine learning?"
    print(f"\nTest query: {test_query}")
    
    query_embedding = embedder.embed_query(test_query)
    results = store.query(
        query_text="",
        query_embedding=query_embedding.tolist(),
        n_results=3
    )
    
    for i in range(len(results['documents'][0])):
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        relevance = 1 - distance
        
        print(f"\nResult {i+1}: [{relevance:.3f}]")
        print(f"  Paper: {str(metadata.get('title', 'N/A'))[:60]}...")
        print(f"  Section: {metadata.get('parent_section_name', 'N/A')}")
        print(f"  DB Section ID: {metadata.get('db_section_id', 'N/A')}")
        if metadata.get('is_split', 0):
            desc = str(metadata.get('subsection_description', ''))[:50]
            print(f"  Subsection {metadata.get('subsection_index', 0)}: {desc}...")
        print(f"  Parent: {metadata.get('parent_word_count', 'N/A')} words")
        
    print("\n" + "="*80)
    print("Index Rebuild Complete!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Build vector index from database (incremental by default)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/papers.db",
        help="Path to papers database"
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default="data/vector_store",
        help="Directory for vector store"
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=3,
        help="Sentences per chunk"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force full rebuild (reset and re-embed everything)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    db_path = project_root / args.db_path
    vector_store_dir = project_root / args.vector_store
    
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        print("Run 02_process_papers.py first")
        return
    
    mode = "FULL REBUILD" if args.force_rebuild else "INCREMENTAL"
        
    # Confirm
    print("="*80)
    print(f"Build Vector Index from Database ({mode})")
    print("="*80)
    print(f"\nThis will:")
    print(f"  1. Read sections from database: {db_path}")
    if args.force_rebuild:
        print(f"  2. Generate chunks for ALL papers")
        print(f"  3. Create embeddings (GPU)")
        print(f"  4. Reset and rebuild vector store: {vector_store_dir}")
    else:
        print(f"  2. Generate chunks for NEW papers only")
        print(f"  3. Create embeddings for new chunks (GPU)")
        print(f"  4. Add to existing vector store: {vector_store_dir}")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
        
    # Build
    try:
        rebuild_index_from_database(
            db_path=str(db_path),
            vector_store_dir=str(vector_store_dir),
            sentences_per_chunk=args.sentences,
            batch_size=args.batch_size,
            force_rebuild=args.force_rebuild
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
