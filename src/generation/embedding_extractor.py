"""
Embedding-Based Context Extractor - Fast relevance filtering using embeddings

Replaces LLM-based extraction (Qwen 3B) with embedding similarity scoring.
~25x faster: processes all parents in <2s instead of ~50s.

Uses the already-loaded sentence-transformers embedder to score sub-chunks
of parent sections by relevance to the query, selecting the most relevant
portions while preserving original text order.
"""

import numpy as np
import logging
import time
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingContextExtractor:
    """
    Fast context extraction using embedding cosine similarity.
    
    Strategy:
    1. Split parent sections into ~250-word sub-chunks (with overlap)
    2. Embed all sub-chunks + query in one fast batch
    3. Score by cosine similarity
    4. Select top sub-chunks up to word budget, in original order
    """
    
    def __init__(self, embedder, sub_chunk_words: int = 250, overlap_words: int = 50):
        """
        Initialize embedding-based extractor.
        
        Args:
            embedder: Already-loaded Embedder instance (sentence-transformers)
            sub_chunk_words: Target words per sub-chunk for scoring
            overlap_words: Word overlap between adjacent sub-chunks
        """
        self.embedder = embedder
        self.sub_chunk_words = sub_chunk_words
        self.overlap_words = overlap_words
    
    def _split_into_sub_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping sub-chunks of approximately sub_chunk_words.
        
        Tries to split at sentence boundaries where possible for cleaner chunks.
        """
        words = text.split()
        
        if len(words) <= self.sub_chunk_words:
            return [text]
        
        chunks = []
        start = 0
        step = self.sub_chunk_words - self.overlap_words
        
        while start < len(words):
            end = min(start + self.sub_chunk_words, len(words))
            chunk = ' '.join(words[start:end])
            
            # Try to end at a sentence boundary (look for . ! ? within last 30 words)
            if end < len(words):
                # Search backwards from end for a sentence-ending punctuation
                search_window = min(30, self.sub_chunk_words // 4)
                for i in range(end - 1, max(end - search_window, start + self.sub_chunk_words // 2), -1):
                    if words[i].endswith(('.', '!', '?', '."', '?"', '!"')):
                        end = i + 1
                        chunk = ' '.join(words[start:end])
                        break
            
            chunks.append(chunk)
            start += max(step, 1)
            
            # Avoid tiny trailing chunks
            if len(words) - start < self.sub_chunk_words // 3:
                # Absorb remaining words into the last chunk
                if start < len(words):
                    last_chunk_start = max(0, start - step)
                    chunks[-1] = ' '.join(words[last_chunk_start:])
                break
        
        return chunks
    
    def extract_relevant_context(
        self,
        parent_text: str,
        query: str,
        max_words: int = 1200,
        min_similarity: float = 0.15
    ) -> str:
        """
        Extract relevant portions from a single parent section.
        
        Args:
            parent_text: Full parent section text
            query: User's search query
            max_words: Maximum words to extract
            min_similarity: Minimum similarity threshold
            
        Returns:
            Extracted relevant text (sub-chunks in original order)
        """
        words = parent_text.split()
        
        # If short enough, return as-is
        if len(words) <= max_words:
            return parent_text
        
        # Split into sub-chunks
        sub_chunks = self._split_into_sub_chunks(parent_text)
        
        if len(sub_chunks) <= 1:
            return parent_text
        
        # Embed query and all sub-chunks in one batch
        query_embedding = self.embedder.embed_query(query)
        chunk_embeddings = self.embedder.embed_text(sub_chunks, show_progress=False)
        
        # Compute cosine similarities (embeddings are already L2-normalised)
        similarities = np.dot(chunk_embeddings, query_embedding)
        
        # Select top chunks up to word budget, maintaining original order
        ranked_indices = np.argsort(similarities)[::-1]
        
        selected_indices = []
        total_words = 0
        
        for idx in ranked_indices:
            sim = similarities[idx]
            
            # Skip chunks below minimum similarity
            if sim < min_similarity and len(selected_indices) > 0:
                break
            
            chunk_word_count = len(sub_chunks[idx].split())
            if total_words + chunk_word_count > max_words and len(selected_indices) > 0:
                break
            
            selected_indices.append(int(idx))
            total_words += chunk_word_count
        
        # Sort by original position to maintain text flow
        selected_indices.sort()
        
        extracted = '\n\n'.join(sub_chunks[i] for i in selected_indices)
        return extracted
    
    def process_query_results(
        self,
        query_results: List[Dict],
        query: str,
        max_context_words: int = 8000
    ) -> List[Dict]:
        """
        Process query results, extracting relevant context from parent sections.
        
        Drop-in replacement for ContextExtractor.process_query_results().
        
        Args:
            query_results: List of dicts with 'parent_text', 'relevance', etc.
            query: Original search query
            max_context_words: Total word budget across all results
            
        Returns:
            Enhanced query results with 'extracted_context' field
        """
        logger.info(f"Extracting context from {len(query_results)} parents (embedding-based)...")
        start = time.time()
        
        # Word budget per parent
        max_per_parent = max_context_words // max(1, len(query_results))
        
        enhanced_results = []
        total_parent_words = 0
        total_extracted_words = 0
        
        for i, result in enumerate(query_results):
            parent_text = result['parent_text']
            parent_word_count = len(parent_text.split())
            total_parent_words += parent_word_count
            
            # Extract relevant portions
            extracted = self.extract_relevant_context(
                parent_text=parent_text,
                query=query,
                max_words=max_per_parent
            )
            
            # Fallback to full text if extraction fails or returns empty
            if not extracted or len(extracted.strip()) < 50:
                logger.warning(f"  Extraction returned empty/short for parent {i+1}, using full parent text")
                extracted = parent_text[:max_per_parent * 6]  # Approximate word->char conversion
            
            extracted_word_count = len(extracted.split())
            total_extracted_words += extracted_word_count
            
            enhanced = result.copy()
            enhanced['extracted_context'] = extracted
            enhanced['extracted_word_count'] = extracted_word_count
            enhanced['compression_ratio'] = parent_word_count / max(1, extracted_word_count)
            enhanced_results.append(enhanced)
        
        elapsed = time.time() - start
        overall_compression = total_parent_words / max(1, total_extracted_words)
        
        logger.info(f"\nContext Extraction Summary (embedding-based):")
        logger.info(f"  Total parent words: {total_parent_words:,}")
        logger.info(f"  Total extracted words: {total_extracted_words:,}")
        logger.info(f"  Compression ratio: {overall_compression:.1f}x")
        logger.info(f"  Time: {elapsed:.2f}s")
        
        return enhanced_results
