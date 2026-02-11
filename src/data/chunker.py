"""
Parent-Document Chunker
Two-tier retrieval system:
- Tier 1: Small 3-sentence chunks for precise vector search
- Tier 2: Full parent sections (30-50 sentences) for LLM context
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ParentDocumentChunker:
    """
    Chunk documents with parent-document tracking.
    
    Search small chunks, return full parent sections.
    """
    
    def __init__(self,
                 sentences_per_chunk: int = 3,
                 overlap_sentences: int = 1,
                 metadata_file: Optional[str] = None):
        """
        Initialize the chunker.
        
        Args:
            sentences_per_chunk: Sentences per search chunk (default: 3)
            overlap_sentences: Overlap between chunks (default: 1)
            metadata_file: Path to arxiv_metadata.json for filling missing metadata
        """
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        
        # Load metadata dictionary for missing fields
        self.metadata_dict = {}
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                # New format: dict with arxiv_id as key
                if isinstance(metadata, dict):
                    self.metadata_dict = metadata
                # Old format: list of papers
                elif isinstance(metadata, list):
                    for item in metadata:
                        arxiv_id = item.get('arxiv_id', '')
                        self.metadata_dict[arxiv_id] = item
            print(f"  Loaded metadata for {len(self.metadata_dict)} papers")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences with proper handling of abbreviations.
        
        Handles:
        - Abbreviations: Dr., Ph.D., et al., i.e., e.g., etc.
        - Decimal numbers: 3.14, 0.5
        - Standard endings: . ! ?
        """
        # Protect abbreviations
        text = re.sub(r'Dr\.', 'Dr<PERIOD>', text)
        text = re.sub(r'Mr\.', 'Mr<PERIOD>', text)
        text = re.sub(r'Mrs\.', 'Mrs<PERIOD>', text)
        text = re.sub(r'Ms\.', 'Ms<PERIOD>', text)
        text = re.sub(r'Prof\.', 'Prof<PERIOD>', text)
        text = re.sub(r'Ph\.D\.', 'PhD<PERIOD>', text)
        text = re.sub(r'et\s+al\.', 'et al<PERIOD>', text)
        text = re.sub(r'i\.e\.', 'ie<PERIOD>', text)
        text = re.sub(r'e\.g\.', 'eg<PERIOD>', text)
        text = re.sub(r'vs\.', 'vs<PERIOD>', text)
        text = re.sub(r'etc\.', 'etc<PERIOD>', text)
        
        # Protect decimal numbers
        text = re.sub(r'(\d)\.(\d)', r'\1<PERIOD>\2', text)
        
        # Split on sentence boundaries
        # Pattern: . ! ? followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore protected periods
        sentences = [s.replace('<PERIOD>', '.') for s in sentences]
        
        # Clean and filter
        sentences = [
            s.strip() 
            for s in sentences 
            if s.strip() and len(s.strip()) > 20
        ]
        
        return sentences
    
    def chunk_section(self,
                     section_text: str,
                     section_name: str,
                     arxiv_id: str) -> List[Dict]:
        """
        Chunk a single section into small chunks while preserving parent.
        
        Args:
            section_text: Text of the section
            section_name: Name of the section (e.g., 'introduction')
            arxiv_id: ArXiv ID of the paper
            
        Returns:
            List of chunk dictionaries with parent tracking
        """
        # Split section into sentences
        sentences = self._split_into_sentences(section_text)
        
        if len(sentences) < self.sentences_per_chunk:
            # Section too small, make it a single chunk
            return [{
                'chunk_id': f"{arxiv_id}_{section_name}_0",
                'chunk_text': section_text,
                'chunk_sentence_count': len(sentences),
                'chunk_word_count': len(section_text.split()),
                'parent_section_id': f"{arxiv_id}_{section_name}",
                'parent_section_name': section_name,
                'parent_text': section_text,
                'parent_sentence_count': len(sentences),
                'parent_word_count': len(section_text.split()),
                'arxiv_id': arxiv_id,
            }]
        
        # Create small chunks for search
        chunks = []
        i = 0
        chunk_index = 0
        
        while i < len(sentences):
            # Get sentences for this chunk
            end_idx = min(i + self.sentences_per_chunk, len(sentences))
            chunk_sentences = sentences[i:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            
            # Create chunk with parent reference
            chunk = {
                'chunk_id': f"{arxiv_id}_{section_name}_{chunk_index}",
                'chunk_text': chunk_text,
                'chunk_sentence_count': len(chunk_sentences),
                'chunk_word_count': len(chunk_text.split()),
                
                # Parent section info
                'parent_section_id': f"{arxiv_id}_{section_name}",
                'parent_section_name': section_name,
                'parent_text': section_text,  # Full section text
                'parent_sentence_count': len(sentences),
                'parent_word_count': len(section_text.split()),
                
                # Paper metadata
                'arxiv_id': arxiv_id,
            }
            
            chunks.append(chunk)
            
            # Move forward with overlap
            step = self.sentences_per_chunk - self.overlap_sentences
            i += max(step, 1)
            chunk_index += 1
        
        return chunks
    
    def chunk_paper(self, paper_data: Dict) -> List[Dict]:
        """
        Chunk an entire paper into searchable chunks with parent sections.
        
        Args:
            paper_data: Paper dictionary with sections
            
        Returns:
            List of all chunks from all sections
        """
        all_chunks = []
        arxiv_id = paper_data['arxiv_id']
        
        # Get paper metadata - check processed file first, then fallback to raw metadata
        title = paper_data.get('title', '') or (self.metadata_dict.get(arxiv_id, {}).get('title', '') if self.metadata_dict else '')
        authors = paper_data.get('authors', []) or (self.metadata_dict.get(arxiv_id, {}).get('authors', []) if self.metadata_dict else [])
        categories = paper_data.get('categories', []) or (self.metadata_dict.get(arxiv_id, {}).get('categories', []) if self.metadata_dict else [])
        published = paper_data.get('published', '') or (self.metadata_dict.get(arxiv_id, {}).get('published', '') if self.metadata_dict else '')
        parsing_method = paper_data.get('parsing_method', '')
        
        # Get sections
        sections = paper_data.get('sections', {})
        
        if not sections:
            # No sections, use full text
            full_text = paper_data.get('full_text', '')
            if full_text:
                chunks = self.chunk_section(
                    full_text,
                    'full_text',
                    arxiv_id
                )
                # Add metadata to chunks
                for chunk in chunks:
                    chunk['title'] = title
                    chunk['authors'] = authors
                    chunk['categories'] = categories
                    chunk['published'] = published
                    chunk['parsing_method'] = parsing_method
                all_chunks.extend(chunks)
        else:
            # Chunk each section
            for section_name, section_text in sections.items():
                chunks = self.chunk_section(
                    section_text,
                    section_name,
                    arxiv_id
                )
                
                # Add paper metadata to each chunk
                for chunk in chunks:
                    chunk['title'] = title
                    chunk['authors'] = authors
                    chunk['categories'] = categories
                    chunk['published'] = published
                    chunk['parsing_method'] = parsing_method
                
                all_chunks.extend(chunks)
        
        return all_chunks