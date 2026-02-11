"""
Generation module - LLM-based components for RAG answer generation.

Components:
- EmbeddingContextExtractor: Extract relevant portions using embedding similarity
- AnswerGenerator: Generate answers with citations using local LLM
- PromptBuilder: Construct prompts for answer generation
"""

from .embedding_extractor import EmbeddingContextExtractor
from .answer_generator import AnswerGenerator
from .prompt_builder import PromptBuilder

__all__ = [
    'EmbeddingContextExtractor',
    'AnswerGenerator', 
    'PromptBuilder'
]