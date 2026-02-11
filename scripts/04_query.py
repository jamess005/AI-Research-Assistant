#!/usr/bin/env python3
"""
Script 04: Query RAG Pipeline

Complete end-to-end query pipeline:
  Query → Vector Search → Context Extraction → Answer Generation

Usage:
    python 04_query.py --query "your question here"
    python 04_query.py --interactive
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.retrieval.vector_store import VectorStore
from src.data.embedder import Embedder
from src.generation.embedding_extractor import EmbeddingContextExtractor
from src.generation.answer_generator import AnswerGenerator


def complete_rag_query(
    query: str,
    vector_store: VectorStore,
    embedder: Embedder,
    context_extractor: Optional[EmbeddingContextExtractor],
    answer_generator_config: Dict,
    k: int = 5,
    save_output: bool = False,
    save_markdown: bool = False,
    answer_generator: Optional[AnswerGenerator] = None,
) -> Dict:
    """
    Execute complete RAG pipeline.

    Pipeline:
        Query → Embed → Vector Search → Context Extraction → Answer Generation
    """
    print("\n" + "=" * 80)
    print("COMPLETE RAG PIPELINE")
    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    pipeline_start = time.time()

    # Stage 1: Vector Search
    print("\n[1/3] Vector Search...")
    search_start = time.time()

    query_embedding = embedder.embed_query(query)

    # Retrieve more chunks initially to ensure k diverse parent sections
    initial_k = k * 5
    raw_results = vector_store.query(
        query_text="",
        query_embedding=query_embedding.tolist(),
        n_results=initial_k
    )

    # Deduplicate by parent section and keep top k unique parents
    results = []
    seen_parents = set()

    for i in range(len(raw_results['documents'][0])):
        metadata = raw_results['metadatas'][0][i]
        parent_key = f"{metadata.get('arxiv_id', '')}_{metadata.get('parent_section_name', '')}"

        if parent_key not in seen_parents:
            seen_parents.add(parent_key)
            results.append({
                'parent_text': raw_results['documents'][0][i],
                'metadata': metadata,
                'distance': raw_results['distances'][0][i],
                'relevance': 1 - raw_results['distances'][0][i]
            })

            if len(results) >= k:
                break

    search_time = time.time() - search_start
    print(f"  Retrieved {len(results)} results in {search_time:.1f}s")

    total_parent_words = sum(len(r['parent_text'].split()) for r in results)
    print(f"  Total parent text: {total_parent_words:,} words")

    # Stage 2: Context Extraction (optional)
    if context_extractor:
        print("\n[2/3] Context Extraction...")
        extraction_start = time.time()

        extracted_contexts = context_extractor.process_query_results(
            query_results=results,
            query=query,
            max_context_words=8000
        )

        extraction_time = time.time() - extraction_start

        total_extracted_words = sum(
            len(ctx['extracted_context'].split())
            for ctx in extracted_contexts
        )
        compression = total_parent_words / max(1, total_extracted_words)

        print(f"  Extracted {total_extracted_words:,} words in {extraction_time:.1f}s")
        print(f"  Compression: {compression:.1f}x")

        print(f"  Context extracted in {extraction_time:.1f}s")
    else:
        # Skip extraction - use raw parent texts
        print("\n[2/3] Context Extraction... SKIPPED")
        extraction_time = 0
        extracted_contexts = [
            {
                'parent_text': r['parent_text'],
                'extracted_context': r['parent_text'][:5000],  # Truncate to 5000 chars
                'metadata': r['metadata'],
                'relevance': r['relevance']
            }
            for r in results
        ]
        print("  Using raw parent sections (truncated to 5000 chars)")

    # Stage 3: Answer Generation
    print("\n[3/3] Answer Generation...")
    
    # Use provided generator or create one
    if answer_generator is None:
        print(f"  Loading answer generator: {answer_generator_config['model_name']}...")
        answer_generator = AnswerGenerator(**answer_generator_config)
        print("  Answer generator ready")
    
    generation_start = time.time()

    result = answer_generator.generate_answer(
        query=query,
        extracted_contexts=extracted_contexts
    )

    generation_time = time.time() - generation_start
    print(f"  Generated answer in {generation_time:.1f}s")

    pipeline_time = time.time() - pipeline_start

    result['timing'] = {
        'search': search_time,
        'extraction': extraction_time,
        'generation': generation_time,
        'total': pipeline_time
    }

    # Display formatted output
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(answer_generator.format_output(result))

    # Timing summary
    print("\n" + "=" * 80)
    print("PIPELINE TIMING")
    print("=" * 80)
    print(f"  Vector search:      {search_time:6.1f}s ({search_time/pipeline_time*100:5.1f}%)")
    print(f"  Context extraction: {extraction_time:6.1f}s ({extraction_time/pipeline_time*100:5.1f}%)")
    print(f"  Answer generation:  {generation_time:6.1f}s ({generation_time/pipeline_time*100:5.1f}%)")
    print(f"  {'─' * 40}")
    print(f"  Total:              {pipeline_time:6.1f}s")

    # Save if requested
    if save_output or save_markdown:
        output_dir = project_root / "data" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_markdown:
            output_file = output_dir / "rag_result.md"
            with open(output_file, 'w') as f:
                f.write(f"# RAG Query Result\n\n")
                f.write(f"**Query:** {result['query']}\n\n---\n\n")
                f.write(f"## Answer\n\n{result['answer']}\n\n---\n\n")
                f.write(f"## Sources\n\n")

                for source in result['sources']:
                    status = "**CITED**" if source['cited'] else "Not cited"
                    f.write(f"### [{source['number']}] {status}\n\n")
                    if source['title'] != 'Unknown':
                        f.write(f"**Title:** {source['title']}\n\n")
                    f.write(f"**Paper:** {source['arxiv_id']} | "
                            f"**Section:** {source['section']} | "
                            f"**Relevance:** {source['relevance']:.2f}\n\n")

                f.write(f"---\n\n## Metadata\n\n")
                f.write(f"- **Model:** {result['metadata']['model']}\n")
                f.write(f"- **Generation time:** {result['generation_time']:.1f}s\n")
                f.write(f"- **Answer length:** {result['metadata']['answer_word_count']} words\n")
                f.write(f"- **Sources cited:** {result['metadata']['num_sources_cited']}"
                        f"/{result['metadata']['num_sources_available']}\n")

            print(f"\nMarkdown saved to: {output_file}")

        if save_output:
            output_file = output_dir / "rag_result.json"
            json_result = {
                'query': result['query'],
                'answer': result['answer'],
                'citations_used': result['citations_used'],
                'sources': result['sources'],
                'timing': result['timing'],
                'metadata': result['metadata']
            }

            with open(output_file, 'w') as f:
                json.dump(json_result, f, indent=2)

            print(f"JSON saved to: {output_file}")

    return result


def interactive_mode(
    vector_store: VectorStore,
    embedder: Embedder,
    context_extractor: Optional[EmbeddingContextExtractor],
    answer_generator: AnswerGenerator,
    answer_generator_config: Dict,
):
    """Run interactive query loop."""
    print("\n" + "=" * 80)
    print("INTERACTIVE RAG MODE")
    print("=" * 80)
    print("Ask questions about ML research. Type 'quit' or 'exit' to stop.")
    print("=" * 80)

    while True:
        try:
            print("\n")
            query = input("Question: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not query:
                continue

            complete_rag_query(
                query=query,
                vector_store=vector_store,
                embedder=embedder,
                context_extractor=context_extractor,
                answer_generator_config=answer_generator_config,
                answer_generator=answer_generator,
                k=5,
            )

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete RAG pipeline with context extraction and answer generation"
    )
    parser.add_argument(
        "--query", type=str,
        help="Question to ask"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of results to retrieve (default: 5)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Save results as Markdown file"
    )
    parser.add_argument(
        "--generation-model", type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model for answer generation (default: Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--no-quantization", action="store_true",
        help="Disable 4-bit quantization (slower, more VRAM)"
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip context extraction (faster but less accurate)"
    )

    args = parser.parse_args()

    if not args.interactive and not args.query:
        parser.error("Either --query or --interactive must be specified")

    print("=" * 80)
    print("INITIALIZING RAG SYSTEM")
    print("=" * 80)

    # Initialize embedder first (needed for vector store queries + context extraction)
    print("\n[1/4] Loading embedder...")
    embedder = Embedder(
        model_name='all-mpnet-base-v2',
        cache_dir=str(project_root / "models" / "embedding"),
        force_gpu=True
    )
    print("  Embedder ready")

    # Initialize vector store
    print("\n[2/4] Loading vector store...")
    vector_store = VectorStore(
        persist_directory=str(project_root / "data" / "vector_store"),
        collection_name="research_papers"
    )
    print(f"  Vector store loaded: {vector_store.collection.count():,} chunks")

    # Initialize context extractor (embedding-based, fast)
    if not args.skip_extraction:
        print("\n[3/4] Initializing context extractor (embedding-based)...")
        context_extractor = EmbeddingContextExtractor(
            embedder=embedder,
            sub_chunk_words=250,
            overlap_words=50
        )
        print("  Context extractor ready")
    else:
        print("\n[3/4] Context extraction: DISABLED (using raw sections)")
        context_extractor = None

    # Load answer generator
    print(f"\n[4/4] Loading answer generator: {args.generation_model}...")
    answer_generator_config = {
        'model_name': args.generation_model,
        'device': 'auto',
        'load_in_4bit': not args.no_quantization,
        'cache_dir': str(project_root / "models" / "llm"),
        'max_new_tokens': 1500
    }
    answer_generator = AnswerGenerator(**answer_generator_config)
    print("  Answer generator ready")

    print("\n" + "=" * 80)
    print("SYSTEM READY")
    print("=" * 80)

    if args.interactive:
        interactive_mode(
            vector_store=vector_store,
            embedder=embedder,
            context_extractor=context_extractor,
            answer_generator=answer_generator,
            answer_generator_config=answer_generator_config,
        )
    else:
        complete_rag_query(
            query=args.query,
            vector_store=vector_store,
            embedder=embedder,
            context_extractor=context_extractor,
            answer_generator_config=answer_generator_config,
            answer_generator=answer_generator,
            k=args.k,
            save_output=args.save,
            save_markdown=args.markdown,
        )


if __name__ == "__main__":
    main()
