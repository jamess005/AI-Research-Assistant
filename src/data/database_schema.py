"""
Database Schema for RAG Pipeline
Replaces JSON files with efficient relational storage
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import json


class PapersDatabase:
    """SQLite database for papers and sections."""
    
    def __init__(self, db_path: str):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        
    def connect(self):
        """Connect to database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self.cursor = self.conn.cursor()
        
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def create_tables(self):
        """Create papers and sections tables."""
        if self.cursor is None or self.conn is None:
            raise RuntimeError("Database not connected")
        
        # Papers table - stores paper metadata
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                arxiv_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,  -- JSON array of authors
                categories TEXT,  -- JSON array of categories
                published TEXT,
                parsing_method TEXT,
                section_count INTEGER,
                word_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Sections table - stores section text with intelligent splitting
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arxiv_id TEXT NOT NULL,
                section_name TEXT NOT NULL,
                section_text TEXT NOT NULL,
                subsection_index INTEGER DEFAULT 0,  -- For split sections
                subsection_description TEXT,  -- Brief description of subsection
                word_count INTEGER,
                sentence_count INTEGER,
                is_split BOOLEAN DEFAULT 0,  -- Whether this was split from larger section
                original_section_name TEXT,  -- If split, the original section name
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id) ON DELETE CASCADE
            )
        """)
        
        # Create indices for fast lookup
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sections_arxiv_id 
            ON sections(arxiv_id)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sections_name 
            ON sections(arxiv_id, section_name)
        """)
        
        self.conn.commit()
        
    def insert_paper(self, paper_data: Dict[str, Any]) -> None:
        """
        Insert paper metadata.
        
        Args:
            paper_data: Dictionary with paper metadata
        """
        if self.cursor is None or self.conn is None:
            raise RuntimeError("Database not connected")
        self.cursor.execute("""
            INSERT OR REPLACE INTO papers 
            (arxiv_id, title, authors, categories, published, parsing_method, 
             section_count, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            paper_data['arxiv_id'],
            paper_data.get('title', ''),
            json.dumps(paper_data.get('authors', [])),
            json.dumps(paper_data.get('categories', [])),
            paper_data.get('published', ''),
            paper_data.get('parsing_method', 'latex'),
            paper_data.get('section_count', 0),
            paper_data.get('word_count', 0)
        ))
        self.conn.commit()
        
    def insert_section(self, section_data: Dict[str, Any]) -> int:
        """
        Insert section.
        
        Args:
            section_data: Dictionary with section data
            
        Returns:
            Section ID
        """
        if self.cursor is None or self.conn is None:
            raise RuntimeError("Database not connected")
        self.cursor.execute("""
            INSERT INTO sections 
            (arxiv_id, section_name, section_text, subsection_index, 
             subsection_description, word_count, sentence_count, 
             is_split, original_section_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            section_data['arxiv_id'],
            section_data['section_name'],
            section_data['section_text'],
            section_data.get('subsection_index', 0),
            section_data.get('subsection_description', ''),
            section_data['word_count'],
            section_data.get('sentence_count', 0),
            section_data.get('is_split', False),
            section_data.get('original_section_name', '')
        ))
        self.conn.commit()
        lastrowid = self.cursor.lastrowid
        if lastrowid is None:
            raise RuntimeError("Failed to insert section - no row ID returned")
        return lastrowid
        
    def get_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Get paper by arxiv_id."""
        if self.cursor is None:
            raise RuntimeError("Database not connected")
        self.cursor.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", 
            (arxiv_id,)
        )
        row = self.cursor.fetchone()
        if row:
            paper = dict(row)
            paper['authors'] = json.loads(paper['authors'])
            paper['categories'] = json.loads(paper['categories'])
            return paper
        return None
        
    def get_sections(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """Get all sections for a paper."""
        if self.cursor is None:
            raise RuntimeError("Database not connected")
        self.cursor.execute(
            "SELECT * FROM sections WHERE arxiv_id = ? ORDER BY id",
            (arxiv_id,)
        )
        return [dict(row) for row in self.cursor.fetchall()]
        
    def get_section_by_id(self, section_id: int) -> Optional[Dict[str, Any]]:
        """Get section by ID."""
        if self.cursor is None:
            raise RuntimeError("Database not connected")
        self.cursor.execute(
            "SELECT * FROM sections WHERE id = ?",
            (section_id,)
        )
        row = self.cursor.fetchone()
        return dict(row) if row else None
        
    def get_all_papers(self) -> List[Dict[str, Any]]:
        """Get all papers."""
        if self.cursor is None:
            raise RuntimeError("Database not connected")
        self.cursor.execute("SELECT * FROM papers ORDER BY arxiv_id")
        papers = []
        for row in self.cursor.fetchall():
            paper = dict(row)
            paper['authors'] = json.loads(paper['authors'])
            paper['categories'] = json.loads(paper['categories'])
            papers.append(paper)
        return papers
        
    def count_papers(self) -> int:
        """Count total papers."""
        if self.cursor is None:
            raise RuntimeError("Database not connected")
        self.cursor.execute("SELECT COUNT(*) FROM papers")
        result = self.cursor.fetchone()
        return result[0] if result else 0
        
    def count_sections(self) -> int:
        """Count total sections."""
        if self.cursor is None:
            raise RuntimeError("Database not connected")
        self.cursor.execute("SELECT COUNT(*) FROM sections")
        result = self.cursor.fetchone()
        return result[0] if result else 0
        
    def get_section_stats(self) -> Dict[str, Any]:
        """Get statistics about sections."""
        if self.cursor is None:
            raise RuntimeError("Database not connected")
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total_sections,
                AVG(word_count) as avg_word_count,
                MAX(word_count) as max_word_count,
                MIN(word_count) as min_word_count,
                SUM(is_split) as split_sections
            FROM sections
        """)
        return dict(self.cursor.fetchone())
