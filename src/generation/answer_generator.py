"""
Answer Generator - Local LLM answer generation with citation tracking
"""

import torch
import re
import time
import logging
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .prompt_builder import PromptBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Generates answers from extracted contexts using local Llama model.
    
    Features:
    - Local inference (no API calls)
    - Numbered citation support
    - Citation tracking and validation
    - ROCm/CUDA/CPU compatible
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-8B-Instruct",
        device: str = "auto",
        load_in_4bit: bool = True,
        max_new_tokens: int = 1500,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the answer generator.
        
        Args:
            model_name: HuggingFace model ID
            device: Device to use ("auto", "cuda", "cpu")
            load_in_4bit: Use 4-bit quantization (recommended)
            max_new_tokens: Maximum tokens to generate (roughly 1000 words)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.prompt_builder = PromptBuilder()
        
        # Detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing AnswerGenerator on {self.device}")
        logger.info(f"Model: {model_name}")
        
        # Configure quantization
        quantization_config = None
        if load_in_4bit and self.device in ["cuda", "rocm"]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Using 4-bit quantization")
        
        # Load model
        logger.info("Loading model... (first time: ~60s for download + initialization)")
        start = time.time()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
        except OSError as e:
            if "gated" in str(e).lower() or "authentication" in str(e).lower() or "401" in str(e):
                logger.error(f"Model {model_name} requires authentication or is gated.")
                logger.error("Solutions:")
                logger.error("  1. Use a different model: --generation-model Qwen/Qwen2.5-7B-Instruct")
                logger.error("  2. Or authenticate: huggingface-cli login")
                raise RuntimeError(f"Model {model_name} requires HuggingFace authentication. Use --generation-model to specify a different model.") from e
            raise
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        try:
            # Use single GPU device map for ROCm compatibility
            device_map_value = {"" : 0} if self.device in ["cuda", "rocm"] else None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map_value,
                dtype=torch.float16 if self.device in ["cuda", "rocm"] else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir
            )
        except OSError as e:
            if "gated" in str(e).lower() or "authentication" in str(e).lower() or "401" in str(e):
                logger.error(f"Model {model_name} requires authentication or is gated.")
                logger.error("Solutions:")
                logger.error("  1. Use a different model: --generation-model Qwen/Qwen2.5-7B-Instruct")
                logger.error("  2. Or authenticate: huggingface-cli login")
                raise RuntimeError(f"Model {model_name} requires HuggingFace authentication. Use --generation-model to specify a different model.") from e
            raise
        
        # Model is already on correct device via device_map parameter
        self.model.eval()
        
        load_time = time.time() - start
        logger.info(f"Model loaded in {load_time:.1f}s")
    
    def cleanup(self):
        """Explicitly clean up model and free GPU memory."""
        logger.info("Cleaning up AnswerGenerator...")
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("AnswerGenerator cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.cleanup()
        except Exception:
            pass
    
    def generate_answer(
        self,
        query: str,
        extracted_contexts: List[Dict],
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict:
        """
        Generate an answer from extracted contexts.
        
        Args:
            query: User's question
            extracted_contexts: List of context dicts with arxiv_id, section, extracted_context
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling threshold
        
        Returns:
            Dict with:
                - answer: Generated text
                - citations_used: List of source numbers cited
                - sources: Formatted source information
                - generation_time: Time taken
                - metadata: Additional info
        """
        logger.info(f"Generating answer for: '{query[:60]}...'")
        start = time.time()
        
        # Build prompt using chat format (Llama 3.2 supports this)
        messages = self.prompt_builder.format_for_chat_model(query, extracted_contexts)
        
        # Format for Llama chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize with larger context window (Llama 3.1 supports 128K)
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=10240  # ~7K words of context, leaving plenty of room for 1500 token answer
        )
        
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        logger.info("  Generating response...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode ONLY the generated tokens (not the input prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        logger.debug(f"  Input tokens: {input_length}, Generated tokens: {len(generated_tokens)}")
        logger.debug(f"  Answer length: {len(answer)} chars, {len(answer.split())} words")
        
        generation_time = time.time() - start
        logger.info(f"Answer generated in {generation_time:.1f}s")
        
        # Clean the answer (remove any stray formatting)
        answer = self._clean_answer(answer)
        
        # Parse citations
        citations_used = self._extract_citations(answer)
        
        # Build source information
        sources = self._build_source_info(extracted_contexts, citations_used)
        
        return {
            'query': query,
            'answer': answer.strip(),
            'citations_used': sorted(citations_used),
            'sources': sources,
            'generation_time': generation_time,
            'metadata': {
                'model': self.model_name,
                'temperature': temperature,
                'answer_word_count': len(answer.split()),
                'num_sources_available': len(extracted_contexts),
                'num_sources_cited': len(citations_used)
            }
        }
    
    def _clean_answer(self, answer: str) -> str:
        """Clean up answer text - remove any stray artifacts."""
        # Remove any leading "assistant" marker that might slip through
        if answer.strip().lower().startswith("assistant"):
            answer = answer.strip()[len("assistant"):].strip()
        
        # Remove any leading special tokens that weren't caught
        answer = answer.replace("<|eot_id|>", "").replace("<|start_header_id|>", "").replace("<|end_header_id|>", "")
        
        # Remove leading colons, commas, periods, or newlines  
        answer = answer.lstrip(':,.;\n\r\t ').strip()
        
        # Convert bullet points to proper format
        answer = re.sub(r'^\* ', '• ', answer, flags=re.MULTILINE)
        answer = re.sub(r'\n\* ', '\n• ', answer)
        
        return answer.strip()
    
    def _extract_citations(self, answer: str) -> List[int]:
        """
        Extract citation numbers from answer text.
        
        Looks for patterns like [1], [2], [1, 2, 3], etc.
        """
        citations = set()
        
        # Pattern: [1], [2, 3], [1,2,3], etc.
        pattern = r'\[(\d+(?:\s*,\s*\d+)*)\]'
        matches = re.findall(pattern, answer)
        
        for match in matches:
            # Split by comma and extract numbers
            nums = [int(n.strip()) for n in match.split(',')]
            citations.update(nums)
        
        return list(citations)
    
    def _build_source_info(
        self,
        contexts: List[Dict],
        citations_used: List[int]
    ) -> List[Dict]:
        """
        Build formatted source information for cited papers.
        
        Returns list of dicts with source details for display.
        """
        sources = []
        
        for i, ctx in enumerate(contexts, 1):
            metadata = ctx.get('metadata', {})
            
            # Handle empty strings (not just missing keys)  
            arxiv_id = metadata.get('arxiv_id') or metadata.get('paper_id') or 'Unknown'
            title = metadata.get('title') or 'Unknown'
            section = metadata.get('parent_section_name') or metadata.get('section') or 'unknown'
            
            authors = metadata.get('authors') or ''
            categories = metadata.get('categories') or ''
            
            source_info = {
                'number': i,
                'arxiv_id': arxiv_id,
                'title': title,
                'section': section,
                'authors': authors,
                'categories': categories,
                'relevance': ctx.get('relevance', 0.0),
                'cited': i in citations_used,
            }
            sources.append(source_info)
        
        return sources
    
    def format_output(self, result: Dict) -> str:
        """
        Format generation result as human-readable text.
        
        Args:
            result: Dict from generate_answer()
        
        Returns:
            Formatted string for display
        """
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("QUESTION:")
        lines.append("=" * 80)
        lines.append(result['query'])
        lines.append("")
        
        # Answer
        lines.append("=" * 80)
        lines.append("ANSWER:")
        lines.append("=" * 80)
        lines.append(result['answer'])
        lines.append("")
        
        # Sources section
        lines.append("=" * 80)
        lines.append("SOURCES:")
        lines.append("=" * 80)
        
        for source in result['sources']:
            status = "CITED" if source['cited'] else "Not cited"
            lines.append(f"\n[{source['number']}] {status}")
            
            title = source['title']
            if title and title != 'Unknown':
                if len(title) > 70:
                    title = title[:67] + "..."
                lines.append(f"    {title}")
            
            lines.append(f"    Paper: {source['arxiv_id']} | Section: {source['section']} | Relevance: {source['relevance']:.2f}")
        
        # Metadata
        meta = result['metadata']
        lines.append("\n" + "=" * 80)
        lines.append("METADATA:")
        lines.append("=" * 80)
        lines.append(f"  Model: {meta['model']}")
        lines.append(f"  Generation time: {result['generation_time']:.1f}s")
        lines.append(f"  Answer length: {meta['answer_word_count']} words")
        lines.append(f"  Sources cited: {meta['num_sources_cited']}/{meta['num_sources_available']}")
        
        return "\n".join(lines)