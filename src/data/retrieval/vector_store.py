"""
Vector Store Handler using ChromaDB
Manages document embeddings and similarity search.
"""

import logging
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any, cast
from pathlib import Path
import json
import sys

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store interface using ChromaDB."""
    
    def __init__(self, 
                 persist_directory: str = "data/vector_store",
                 collection_name: str = "research_papers",
                 embedder: Optional[Any] = None):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the ChromaDB data
            collection_name: Name of the collection
            embedder: Optional Embedder instance for query-time embedding.
                      If not provided, queries without pre-computed embeddings
                      will raise an error (to prevent ChromaDB's default 
                      384-dim MiniLM from conflicting with stored 768-dim 
                      mpnet embeddings).
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Get or create collection
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection '{self.collection_name}'")
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection '{self.collection_name}'")
        
        return collection
    
    def add_documents(self,
                     documents: List[str],
                     metadatas: List[Dict[str, Any]],
                     ids: List[str],
                     embeddings: Optional[List[List[float]]] = None):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of text chunks
            metadatas: List of metadata dicts for each chunk
            ids: List of unique IDs for each chunk
            embeddings: Pre-computed embeddings (optional, will compute if not provided)
        """
        if embeddings is not None:
            # Add with pre-computed embeddings
            self.collection.add(
                documents=documents,
                metadatas=cast(Any, metadatas),
                ids=ids,
                embeddings=cast(Any, embeddings)
            )
        else:
            # ChromaDB will compute embeddings using default model
            self.collection.add(
                documents=documents,
                metadatas=cast(Any, metadatas),
                ids=ids
            )
        
        logger.info(f"Added {len(documents)} documents to collection")
    
    def query(self,
              query_text: str,
              n_results: int = 5,
              where: Optional[Dict] = None,
              query_embedding: Optional[List[float]] = None) -> Any:
        """
        Query the vector store for similar documents.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            where: Optional metadata filter
            query_embedding: Pre-computed query embedding (768-dim mpnet)
            
        Returns:
            Dictionary with ids, documents, metadatas, distances
            
        Raises:
            ValueError: If no query_embedding provided and no embedder configured
        """
        if query_embedding is not None:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
        elif self.embedder is not None:
            # Use the configured embedder (mpnet, 768-dim) instead of
            # ChromaDB's default MiniLM (384-dim)
            embedding = self.embedder.embed_query(query_text)
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=n_results,
                where=where
            )
        else:
            raise ValueError(
                "No query_embedding provided and no embedder configured. "
                "Either pass query_embedding (768-dim mpnet) or initialise "
                "VectorStore with embedder=Embedder() to avoid ChromaDB's "
                "default 384-dim MiniLM causing a dimension mismatch."
            )
        
        return results
    
    def get_existing_arxiv_ids(self) -> set:
        """Get the set of unique arxiv_ids already in the collection."""
        count = self.count()
        if count == 0:
            return set()
        
        # ChromaDB get() with no IDs returns all; use include to only get metadata
        # Process in pages to avoid memory issues with very large collections
        page_size = 10000
        all_ids = set()
        offset = 0
        
        while offset < count:
            results = self.collection.get(
                limit=page_size,
                offset=offset,
                include=["metadatas"]
            )
            for meta in results.get('metadatas', []):
                if meta and 'arxiv_id' in meta:
                    all_ids.add(meta['arxiv_id'])
            
            batch_size = len(results.get('ids', []))
            if batch_size == 0:
                break
            offset += batch_size
        
        return all_ids

    def get_by_id(self, ids: List[str]) -> Any:
        """Get documents by their IDs."""
        return self.collection.get(ids=ids)
    
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Deleted collection '{self.collection_name}'")
    
    def reset_collection(self):
        """Delete and recreate the collection."""
        self.delete_collection()
        self.collection = self._get_or_create_collection()
        logger.info("Collection reset")
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        count = self.count()
        return {
            "name": self.collection_name,
            "count": count,
            "persist_directory": str(self.persist_directory)
        }