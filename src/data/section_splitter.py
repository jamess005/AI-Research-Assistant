"""
Intelligent Section Splitter - Uses LLM to split large sections intelligently

Splits sections > 5000 words into semantic sub-sections with brief descriptions
for better retrieval and reduced model lag.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import re

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    logger.warning("transformers not installed. Install with: pip install transformers torch")
    torch = None
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    HAS_TRANSFORMERS = False


class SectionSplitter:
    """Intelligently split large sections using LLM."""
    
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        max_section_words: int = 5000,
        device: str = "auto"
    ):
        """
        Initialize section splitter.
        
        Args:
            model_name: HuggingFace model to use for splitting
            max_section_words: Maximum words per section before splitting
            device: Device to run model on
        """
        self.max_section_words = max_section_words
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load LLM model for intelligent splitting."""
        if not HAS_TRANSFORMERS or torch is None:
            raise ImportError("transformers not installed")
            
        logger.info(f"Loading model: {self.model_name}")
        
        # Determine device
        if self.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = self.device
            
        logger.info(f"  Device: {device}")
        
        # Load tokenizer
        if AutoTokenizer is None:
            raise ImportError("transformers not installed")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(project_root / "models" / "llm")
        )
        
        # Load model
        if AutoModelForCausalLM is None:
            raise ImportError("transformers not installed")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(project_root / "models" / "llm"),
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        
        logger.info("Model loaded")
        
    def unload_model(self):
        """Unload model to free memory."""
        if self.model:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def needs_splitting(self, text: str) -> bool:
        """Check if section needs splitting."""
        word_count = len(text.split())
        return word_count > self.max_section_words
        
    def split_section_simple(
        self, 
        section_text: str, 
        target_size: int = 4000
    ) -> List[Tuple[str, str]]:
        """
        Simple sentence-boundary splitting without LLM.
        Falls back to this if LLM is not available.
        
        Returns:
            List of (subsection_text, description) tuples
        """
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', section_text)
        
        subsections = []
        current_subsection = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_word_count + sentence_words > target_size and current_subsection:
                # Finalize current subsection
                subsection_text = ' '.join(current_subsection)
                # Use first sentence as description
                description = current_subsection[0][:200] + "..."
                subsections.append((subsection_text, description))
                
                # Start new subsection
                current_subsection = [sentence]
                current_word_count = sentence_words
            else:
                current_subsection.append(sentence)
                current_word_count += sentence_words
                
        # Add final subsection
        if current_subsection:
            subsection_text = ' '.join(current_subsection)
            description = current_subsection[0][:200] + "..."
            subsections.append((subsection_text, description))
            
        return subsections
        
    def split_section_intelligent(
        self, 
        section_text: str,
        section_name: str,
        paper_title: str = ""
    ) -> List[Tuple[str, str]]:
        """
        Split section using LLM to identify semantic boundaries.
        
        Args:
            section_text: Text to split
            section_name: Name of section
            paper_title: Paper title for context
            
        Returns:
            List of (subsection_text, description) tuples
        """
        if not self.model:
            logger.info("Model not loaded, using simple splitting...")
            return self.split_section_simple(section_text)
            
        # First, do a simple split to get manageable chunks
        # LLM will then analyze each chunk and provide semantic descriptions
        simple_splits = self.split_section_simple(section_text, target_size=4000)
        
        # Use LLM to generate better descriptions for each chunk
        enhanced_splits = []
        
        for i, (chunk_text, _) in enumerate(simple_splits):
            # Create prompt for LLM
            prompt = self._create_description_prompt(
                chunk_text, section_name, paper_title, i + 1, len(simple_splits)
            )
            
            # Generate description
            description = self._generate_description(prompt)
            
            enhanced_splits.append((chunk_text, description))
            
        return enhanced_splits
        
    def _create_description_prompt(
        self, 
        text: str, 
        section_name: str,
        paper_title: str,
        part_num: int,
        total_parts: int
    ) -> str:
        """Create prompt for LLM to generate subsection description."""
        
        # Truncate text for prompt (first 500 words)
        words = text.split()
        preview_text = ' '.join(words[:500])
        if len(words) > 500:
            preview_text += "..."
            
        prompt = f"""You are analyzing a section from a research paper.

Paper Title: {paper_title}
Section: {section_name} (Part {part_num}/{total_parts})

Text Preview:
{preview_text}

Task: Generate a brief, informative description (1-2 sentences, max 30 words) that captures the main topic or contribution of this subsection. Focus on what makes this part unique compared to other parts of the section.

Description:"""

        return prompt
        
    def _generate_description(self, prompt: str) -> str:
        """Generate description using LLM."""
        if self.model is None or self.tokenizer is None:
            return "Section subsection"
            
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            if torch is None:
                return "Section subsection"
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            # Decode
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the description (after "Description:")
            if "Description:" in full_output:
                description = full_output.split("Description:")[-1].strip()
            else:
                description = full_output[len(prompt):].strip()
                
            # Clean up
            description = description.split('\n')[0]  # Take first line
            description = description[:200]  # Truncate if too long
            
            return description if description else "Continuation of section analysis"
            
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return "Section subsection"
            
    def split_section(
        self,
        section_text: str,
        section_name: str,
        paper_title: str = "",
        use_llm: bool = True
    ) -> List[Tuple[str, str]]:
        """
        Main method to split a section.
        
        Args:
            section_text: Text to split
            section_name: Section name
            paper_title: Paper title for context
            use_llm: Whether to use LLM for intelligent splitting
            
        Returns:
            List of (subsection_text, description) tuples
        """
        if not self.needs_splitting(section_text):
            # No splitting needed, return as-is
            return [(section_text, f"Full {section_name} section")]
            
        logger.info(f"Section '{section_name}' has {len(section_text.split())} words, splitting...")
        
        if use_llm and self.model:
            return self.split_section_intelligent(section_text, section_name, paper_title)
        else:
            return self.split_section_simple(section_text)
