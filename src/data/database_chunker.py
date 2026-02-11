"""
Database-Aware Chunker
Chunks sections from database instead of JSON files
"""

import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.database_schema import PapersDatabase
from src.data.chunker import ParentDocumentChunker


class DatabaseChunker:
    """Chunk documents from database."""
    
    def __init__(
        self,
        db_path: str,
        sentences_per_chunk: int = 3,
        overlap_sentences: int = 0
    ):
        """
        Initialize database chunker.
        
        Args:
            db_path: Path to SQLite database
            sentences_per_chunk: Sentences per chunk
            overlap_sentences: Overlap between chunks
        """
        self.db = PapersDatabase(db_path)
        self.chunker = ParentDocumentChunker(
            sentences_per_chunk=sentences_per_chunk,
            overlap_sentences=overlap_sentences
        )
        
    def chunk_paper(self, arxiv_id: str) -> List[Dict]:
        """
        Chunk all sections of a paper.
        
        Args:
            arxiv_id: ArXiv ID of paper
            
        Returns:
            List of chunks with parent section metadata
        """
        # Get paper metadata
        paper = self.db.get_paper(arxiv_id)
        if not paper:
            return []
            
        # Get sections
        sections = self.db.get_sections(arxiv_id)
        
        all_chunks = []
        for section in sections:
            # Chunk this section
            section_chunks = self.chunker.chunk_section(
                section_text=section['section_text'],
                section_name=section['section_name'],
                arxiv_id=arxiv_id
            )
            
            # Add database section metadata to each chunk
            for chunk in section_chunks:
                # Add database-specific fields
                chunk['db_section_id'] = section['id']
                chunk['is_split'] = section['is_split']
                chunk['subsection_index'] = section['subsection_index']
                chunk['subsection_description'] = section['subsection_description']
                chunk['original_section_name'] = section['original_section_name']
                
                # Override parent text with database section text
                # (in case it was split)
                chunk['parent_text'] = section['section_text']
                chunk['parent_section_name'] = section['section_name']
                chunk['parent_word_count'] = section['word_count']
                
                # Add paper metadata (convert lists to strings for ChromaDB compatibility)
                chunk['title'] = paper['title']
                chunk['authors'] = ', '.join(paper['authors']) if isinstance(paper['authors'], list) else paper['authors']
                chunk['categories'] = ', '.join(paper['categories']) if isinstance(paper['categories'], list) else paper['categories']
                chunk['parsing_method'] = paper['parsing_method']
                
                all_chunks.append(chunk)
                
        return all_chunks
        
    def chunk_papers(self, arxiv_ids: List[str], show_progress: bool = True) -> List[Dict]:
        """
        Chunk specific papers by arxiv_id.
        
        Args:
            arxiv_ids: List of arxiv IDs to chunk
            show_progress: Show progress bar
            
        Returns:
            List of chunks for those papers only
        """
        all_chunks = []
        papers_iter = tqdm(arxiv_ids, desc="Chunking new papers") if show_progress else arxiv_ids
        
        for arxiv_id in papers_iter:
            paper_chunks = self.chunk_paper(arxiv_id)
            all_chunks.extend(paper_chunks)
            
        return all_chunks

    def chunk_all_papers(self, show_progress: bool = True) -> List[Dict]:
        """
        Chunk all papers in database.
        
        Args:
            show_progress: Show progress bar
            
        Returns:
            List of all chunks
        """
        papers = self.db.get_all_papers()
        
        all_chunks = []
        papers_iter = tqdm(papers, desc="Chunking papers") if show_progress else papers
        
        for paper in papers_iter:
            paper_chunks = self.chunk_paper(paper['arxiv_id'])
            all_chunks.extend(paper_chunks)
            
        return all_chunks
        
    def get_chunk_statistics(self) -> Dict:
        """Get statistics about generated chunks."""
        all_chunks = self.chunk_all_papers(show_progress=False)
        
        if not all_chunks:
            return {}
            
        chunk_word_counts = [c['chunk_word_count'] for c in all_chunks]
        parent_word_counts = [c['parent_word_count'] for c in all_chunks]
        
        # Count unique parent sections
        unique_parents = set(c['parent_section_id'] for c in all_chunks)
        
        return {
            'total_chunks': len(all_chunks),
            'total_parent_sections': len(unique_parents),
            'avg_chunk_words': sum(chunk_word_counts) / len(chunk_word_counts),
            'avg_parent_words': sum(parent_word_counts) / len(parent_word_counts),
            'min_chunk_words': min(chunk_word_counts),
            'max_chunk_words': max(chunk_word_counts),
            'min_parent_words': min(parent_word_counts),
            'max_parent_words': max(parent_word_counts),
        }
        
    def close(self):
        """Close database connection."""
        self.db.close()
        
    def __enter__(self):
        """Context manager entry."""
        self.db.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
