"""
FastAPI Backend for AI Research Assistant
Wraps the complete RAG pipeline with streaming support.
"""

import os
import sys
import json
import time
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.data.retrieval.vector_store import VectorStore
from src.data.embedder import Embedder
from src.data.database_schema import PapersDatabase
from src.generation.embedding_extractor import EmbeddingContextExtractor
from src.generation.answer_generator import AnswerGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global state for models (loaded once on startup, reused across queries)
class RAGState:
    vector_store: Optional[VectorStore] = None
    embedder: Optional[Embedder] = None
    context_extractor: Optional[EmbeddingContextExtractor] = None
    answer_generator: Optional[AnswerGenerator] = None

rag_state = RAGState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG system on startup and cleanup on shutdown."""
    logger.info("Initializing AI Research Assistant")
    
    try:
        logger.info("[1/4] Loading embedder...")
        rag_state.embedder = Embedder(
            model_name=os.environ.get('EMBEDDING_MODEL_NAME', 'all-mpnet-base-v2'),
            cache_dir=str(project_root / os.environ.get('EMBEDDING_CACHE_DIR', 'models/embedding')),
            force_gpu=True
        )
        
        logger.info("[2/4] Loading vector store...")
        vector_store_dir = project_root / "data" / "vector_store"
        rag_state.vector_store = VectorStore(
            persist_directory=str(vector_store_dir),
            collection_name="research_papers"
        )
        count = rag_state.vector_store.collection.count()
        logger.info(f"  Vector store: {count:,} chunks")
        
        logger.info("[3/4] Initializing context extractor (embedding-based)...")
        rag_state.context_extractor = EmbeddingContextExtractor(
            embedder=rag_state.embedder,
            sub_chunk_words=250,
            overlap_words=50
        )
        
        logger.info("[4/4] Loading answer generation model (Llama 8B)...")
        rag_state.answer_generator = AnswerGenerator(
            model_name=os.environ.get('LLM_MODEL_NAME', 'meta-llama/Llama-3.1-8B-Instruct'),
            device='auto',
            load_in_4bit=True,
            max_new_tokens=1500,
            cache_dir=str(project_root / os.environ.get('LLM_CACHE_DIR', 'models/llm'))
        )
        
        logger.info("AI Research Assistant ready")
        
    except Exception as e:
        logger.exception(f"Startup error: {e}")
        raise
    
    yield  # Server runs here
    
    # Cleanup on shutdown
    logger.info("Shutting down...")
    if rag_state.answer_generator:
        rag_state.answer_generator.cleanup()


# Initialize FastAPI app with lifespan
app = FastAPI(title="AI Research Assistant API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    k: int = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list
    timing: Dict
    metadata: Dict


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    html_file = project_root / "frontend" / "index.html"
    with open(html_file, 'r') as f:
        return f.read()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Count papers from database
    db_path = project_root / "data" / "papers.db"
    paper_count = 0
    section_count = 0
    if db_path.exists():
        with PapersDatabase(str(db_path)) as db:
            paper_count = db.count_papers()
            section_count = db.count_sections()
    
    return {
        "status": "healthy",
        "paper_count": paper_count,
        "section_count": section_count,
        "vector_store_count": rag_state.vector_store.collection.count() if rag_state.vector_store else 0,
        "models_loaded": {
            "vector_store": rag_state.vector_store is not None,
            "embedder": rag_state.embedder is not None
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Execute RAG query using persistent models (no per-query loading)
    """
    if not rag_state.vector_store or not rag_state.embedder:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    if not rag_state.context_extractor or not rag_state.answer_generator:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    pipeline_start = time.time()
    
    try:
        # Stage 1: Vector Search
        search_start = time.time()
        query_embedding = rag_state.embedder.embed_query(request.query)
        
        initial_k = request.k * 5
        raw_results = rag_state.vector_store.query(
            query_text="",
            query_embedding=query_embedding.tolist(),
            n_results=initial_k
        )
        
        # Deduplicate by parent section
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
                
                if len(results) >= request.k:
                    break
        
        search_time = time.time() - search_start
        
        # Stage 2: Context Extraction (using persistent model)
        extraction_start = time.time()
        extracted_contexts = rag_state.context_extractor.process_query_results(
            query_results=results,
            query=request.query,
            max_context_words=5000  # Reduced to ensure prompt fits within token limits
        )
        extraction_time = time.time() - extraction_start
        
        # Stage 3: Answer Generation (using persistent model)
        generation_start = time.time()
        result = rag_state.answer_generator.generate_answer(
            query=request.query,
            extracted_contexts=extracted_contexts
        )
        generation_time = time.time() - generation_start
        pipeline_time = time.time() - pipeline_start
        
        # Build response
        return QueryResponse(
            query=result['query'],
            answer=result['answer'],
            sources=result['sources'],
            timing={
                'search': search_time,
                'extraction': extraction_time,
                'generation': generation_time,
                'total': pipeline_time
            },
            metadata=result['metadata']
        )
        
    except Exception as e:
        logger.exception(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/stream")
async def query_stream(query: str, k: int = 5):
    """
    Stream RAG pipeline progress updates (using persistent models)
    """
    if not rag_state.vector_store or not rag_state.embedder:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    if not rag_state.context_extractor or not rag_state.answer_generator:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Type narrowing for nested function
    vector_store = rag_state.vector_store
    embedder = rag_state.embedder
    context_extractor = rag_state.context_extractor
    answer_generator = rag_state.answer_generator
    
    async def event_generator():
        """Generate Server-Sent Events for pipeline progress"""
        try:
            yield f"data: {json.dumps({'stage': 'start', 'message': 'Starting RAG pipeline...'})}\n\n"
            
            # Stage 1: Vector Search
            yield f"data: {json.dumps({'stage': 'search', 'status': 'running'})}\n\n"
            await asyncio.sleep(0.05)
            
            search_start = time.time()
            query_embedding = embedder.embed_query(query)
            
            initial_k = k * 5
            raw_results = vector_store.query(
                query_text="",
                query_embedding=query_embedding.tolist(),
                n_results=initial_k
            )
            
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
            yield f"data: {json.dumps({'stage': 'search', 'status': 'complete', 'time': search_time, 'results': len(results)})}\n\n"
            
            # Stage 2: Context Extraction (persistent model - no loading delay)
            yield f"data: {json.dumps({'stage': 'extraction', 'status': 'running'})}\n\n"
            await asyncio.sleep(0.05)
            
            extraction_start = time.time()
            extracted_contexts = context_extractor.process_query_results(
                query_results=results,
                query=query,
                max_context_words=5000  # Reduced to ensure prompt fits within token limits
            )
            extraction_time = time.time() - extraction_start
            
            yield f"data: {json.dumps({'stage': 'extraction', 'status': 'complete', 'time': extraction_time})}\n\n"
            
            # Stage 3: Answer Generation (persistent model - no loading delay)
            yield f"data: {json.dumps({'stage': 'generation', 'status': 'running'})}\n\n"
            await asyncio.sleep(0.05)
            
            generation_start = time.time()
            result = answer_generator.generate_answer(
                query=query,
                extracted_contexts=extracted_contexts
            )
            generation_time = time.time() - generation_start
            
            yield f"data: {json.dumps({'stage': 'generation', 'status': 'complete', 'time': generation_time})}\n\n"
            
            # Add timing info to result
            pipeline_time = search_time + extraction_time + generation_time
            result['timing'] = {
                'search': search_time,
                'extraction': extraction_time,
                'generation': generation_time,
                'total': pipeline_time
            }
            
            # Final result
            yield f"data: {json.dumps({'stage': 'complete', 'result': result})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'stage': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/search/papers")
async def search_papers(query: str, k: int = 9):
    """
    Search for relevant papers and return metadata with AI-generated summaries.
    Streams results progressively via SSE.
    """
    if not rag_state.vector_store or not rag_state.embedder:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    if not rag_state.answer_generator:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    vector_store = rag_state.vector_store
    embedder = rag_state.embedder
    answer_generator = rag_state.answer_generator
    db_path = str(project_root / "data" / "papers.db")
    
    async def event_generator():
        try:
            yield f"data: {json.dumps({'stage': 'start', 'message': 'Searching papers...'})}\n\n"
            
            # Stage 1: Vector search, deduplicate by paper_id
            search_start = time.time()
            query_embedding = embedder.embed_query(query)
            
            # Get more results to find enough unique papers
            raw_results = vector_store.query(
                query_text="",
                query_embedding=query_embedding.tolist(),
                n_results=k * 10
            )
            
            # Deduplicate by paper_id, keep best relevance per paper
            seen_papers = {}
            for i in range(len(raw_results['documents'][0])):
                metadata = raw_results['metadatas'][0][i]
                paper_id = metadata.get('arxiv_id', metadata.get('paper_id', ''))
                distance = raw_results['distances'][0][i]
                relevance = 1 - distance
                
                if paper_id not in seen_papers or relevance > seen_papers[paper_id]['relevance']:
                    seen_papers[paper_id] = {
                        'arxiv_id': paper_id,
                        'relevance': relevance,
                        'chunk_text': raw_results['documents'][0][i],
                        'metadata': metadata
                    }
                
                if len(seen_papers) >= k:
                    break
            
            # Sort by relevance
            unique_papers = sorted(seen_papers.values(), key=lambda x: x['relevance'], reverse=True)[:k]
            search_time = time.time() - search_start
            
            yield f"data: {json.dumps({'stage': 'search', 'status': 'complete', 'time': search_time, 'results': len(unique_papers)})}\n\n"
            
            # Stage 2: Look up full metadata from database and generate summaries
            yield f"data: {json.dumps({'stage': 'summarising', 'status': 'running', 'total': len(unique_papers)})}\n\n"
            await asyncio.sleep(0.05)
            
            papers_out = []
            with PapersDatabase(db_path) as db:
                for idx, paper_info in enumerate(unique_papers):
                    paper_id = paper_info['arxiv_id']
                    
                    # Get metadata from database
                    db_paper = db.get_paper(paper_id)
                    
                    if db_paper:
                        title = db_paper.get('title', '') or ''
                        authors = db_paper.get('authors', []) or []
                        published = db_paper.get('published', '') or ''
                        categories = db_paper.get('categories', []) or []
                        word_count = db_paper.get('word_count', 0) or 0
                        
                        # Supplement missing DB fields from vector store metadata
                        meta = paper_info['metadata']
                        if not title or title == 'Unknown':
                            title = meta.get('title', 'Unknown') or 'Unknown'
                        if not authors:
                            authors_str = meta.get('authors', '')
                            if authors_str:
                                authors = [a.strip() for a in authors_str.split(',')]
                    else:
                        # Fallback to vector store metadata
                        meta = paper_info['metadata']
                        title = meta.get('title', 'Unknown')
                        authors_str = meta.get('authors', '')
                        authors = [a.strip() for a in authors_str.split(',')] if authors_str else []
                        published = ''
                        categories = [meta.get('categories', '')]
                        word_count = 0
                    
                    # Get introduction/first section for summarisation
                    sections = db.get_sections(paper_id) if db_paper else []
                    intro_text = ''
                    for sec in sections:
                        sec_name = sec.get('section_name', '').lower()
                        if any(kw in sec_name for kw in ['abstract', 'introduction', 'intro']):
                            intro_text = sec.get('section_text', '')
                            break
                    if not intro_text and sections:
                        intro_text = sections[0].get('section_text', '')
                    
                    # Generate short summary using Llama 8B
                    summary = _generate_paper_summary(
                        answer_generator, title, intro_text, query
                    )
                    
                    paper_result = {
                        'number': idx + 1,
                        'arxiv_id': paper_id,
                        'title': title,
                        'authors': authors,
                        'published': published,
                        'categories': categories,
                        'relevance': paper_info['relevance'],
                        'word_count': word_count,
                        'summary': summary
                    }
                    papers_out.append(paper_result)
                    
                    # Stream each paper as it's ready
                    yield f"data: {json.dumps({'stage': 'paper', 'paper': paper_result, 'progress': idx + 1, 'total': len(unique_papers)})}\n\n"
                    await asyncio.sleep(0.05)
            
            total_time = time.time() - search_start
            yield f"data: {json.dumps({'stage': 'complete', 'papers': papers_out, 'timing': {'search': search_time, 'total': total_time}})}\n\n"
            
        except Exception as e:
            logger.exception(f"Paper search error: {e}")
            yield f"data: {json.dumps({'stage': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _generate_paper_summary(
    answer_generator: AnswerGenerator,
    title: str,
    intro_text: str,
    query: str
) -> str:
    """Generate a short 2-3 sentence summary of a paper using the loaded Llama model."""
    import torch
    import re
    
    # Truncate intro to keep prompts small and fast
    words = intro_text.split()
    if len(words) > 400:
        intro_text = ' '.join(words[:400])
    
    messages = [
        {
            "role": "system",
            "content": (
                "Summarise research papers in exactly 2-3 sentences. "
                "Be specific about the methods, key findings, and contributions. "
                "Write the summary directly with no preamble, no introductory phrases, "
                "and no labels. Do not start with 'This paper', 'The paper', "
                "'The authors', 'Here is', or 'Summary:'. "
                "Start directly with the key contribution or method."
            )
        },
        {
            "role": "user",
            "content": f"Title: {title}\n\nExcerpt:\n{intro_text}"
        }
    ]
    
    prompt = answer_generator.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = answer_generator.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    )
    
    if answer_generator.device != "cpu":
        inputs = {k: v.to(answer_generator.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = answer_generator.model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=answer_generator.tokenizer.pad_token_id,
            eos_token_id=answer_generator.tokenizer.eos_token_id
        )
    
    # Decode only generated tokens
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    summary = answer_generator.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Clean up any stray formatting / special tokens
    summary = summary.replace('<|eot_id|>', '').replace('<|start_header_id|>', '').strip()
    
    # Strip prompt leakage patterns (preamble the model sometimes adds)
    leakage_patterns = [
        r'^Here\s+is\s+a?\s*\d*-?\d*\s*sentence\s+summary[^:]*:\s*',
        r'^Summary[:\s]*',
        r'^In\s+summary[,:\s]*',
        r'^Here\s+is\s+(the|a)\s+summary[^:]*:\s*',
        r'^The\s+summary\s+is[:\s]*',
        r'^\*\*Summary\*\*[:\s]*',
    ]
    for pattern in leakage_patterns:
        summary = re.sub(pattern, '', summary, flags=re.IGNORECASE).strip()
    
    # Remove leading quotes or dashes
    summary = re.sub(r'^["\-\u2013\u2014]+\s*', '', summary).strip()
    
    return summary


# Mount static files
app.mount("/static", StaticFiles(directory=str(project_root / "frontend")), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )