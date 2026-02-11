"""
Embedder Module
Generate embeddings for text chunks using sentence-transformers.
"""

import logging
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union, Optional
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings using sentence-transformers models."""
    
    # Recommended models by size/quality
    MODELS = {
        'small': 'all-MiniLM-L6-v2',        # 384 dims, 80MB, fast
        'medium': 'all-mpnet-base-v2',      # 768 dims, 420MB, balanced
        'large': 'BAAI/bge-large-en-v1.5',  # 1024 dims, 1.34GB, high quality
    }
    
    def __init__(self, 
                 model_name: str = 'all-mpnet-base-v2',
                 device: Optional[str] = None,
                 cache_dir: str = 'models/embedding',
                 force_gpu: bool = True):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            cache_dir: Directory to cache the model
            force_gpu: Force GPU usage if available (default: True)
        """
        # Set up device
        if device is None:
            if force_gpu and torch.cuda.is_available():
                self.device = 'cuda'
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                if force_gpu and not torch.cuda.is_available():
                    logger.warning("GPU requested but not available, using CPU")
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Set up cache directory
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder=str(cache_path)
        )
        
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded (embedding dim: {self.embedding_dim})")
    
    def embed_text(self, text: Union[str, List[str]], 
                   batch_size: int = 32,
                   show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for text.
        
        Args:
            text: Single text string or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        # Convert single string to list
        if isinstance(text, str):
            text = [text]
        
        # Generate embeddings
        embeddings = self.model.encode(
            text,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        embedding = self.embed_text(query, show_progress=False)
        return embedding[0]  # Return single vector
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'max_seq_length': self.model.max_seq_length,
        }