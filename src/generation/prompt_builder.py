"""
Prompt Builder - Constructs prompts for RAG answer generation with citation support
"""

from typing import List, Dict


class PromptBuilder:
    """
    Builds prompts for answer generation with numbered citation support.
    
    Format:
        [1] Paper ID: 2602.05298v1, Section: introduction
        [2] Paper ID: 2602.04998v1, Section: related work
        ...
    
    Answer should cite sources as [1], [2], etc.
    """
    
    def __init__(self):
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt that defines assistant behavior."""
        return """You are a research assistant answering questions about machine learning research.

RULES:
- Organize your answer BY THEME, not by source. Group related ideas from different papers together.
- WRONG: "Source [1] discusses X. Source [2] explores Y. Source [3] proposes Z."
- RIGHT: "X is a key concept [1, 3], which has been extended to Y [2, 4]."
- Each paragraph MUST cite at least 2 different sources.
- Write exactly 3 paragraphs. No more, no less.
- Cite at the end of sentences: [1], [2, 3], etc.
- Start directly with the answer. No preamble or introductions like "Based on the sources".
- Only use information from the provided sources."""
    
    def build_user_prompt(
        self,
        query: str,
        extracted_contexts: List[Dict]
    ) -> str:
        """
        Build the user prompt with query and formatted sources.
        
        Args:
            query: User's question
            extracted_contexts: List of dicts with 'arxiv_id', 'section', 
                               'extracted_context', 'relevance'
        
        Returns:
            Formatted prompt string
        """
        # Format sources with numbers
        sources_section = self._format_sources(extracted_contexts)
        
        prompt = f"""Question: {query}

{sources_section}

Write exactly 3 paragraphs organized by theme, not by source. Each paragraph must cite multiple sources."""
        
        return prompt
    
    def _format_sources(self, contexts: List[Dict]) -> str:
        """Format extracted contexts as numbered sources."""
        formatted_sources = ["SOURCES:"]
        formatted_sources.append("=" * 80)
        
        for i, ctx in enumerate(contexts, 1):
            # Extract metadata properly, handling empty strings
            metadata = ctx.get('metadata', {})
            arxiv_id = metadata.get('arxiv_id') or metadata.get('paper_id') or 'Unknown'
            section = metadata.get('parent_section_name') or metadata.get('section') or 'unknown'
            relevance = ctx.get('relevance', 0.0)
            content = ctx.get('extracted_context', '')
            
            # Truncate very long contexts (shouldn't happen but safety check)
            word_count = len(content.split())
            if word_count > 2000:
                words = content.split()[:2000]
                content = ' '.join(words) + "\n\n[... content truncated ...]"
            
            source_block = f"""
[{i}] Paper ID: {arxiv_id} | Section: {section} | Relevance: {relevance:.2f}
{'-' * 80}
{content}
"""
            formatted_sources.append(source_block)
        
        return "\n".join(formatted_sources)
    
    def build_full_prompt(
        self,
        query: str,
        extracted_contexts: List[Dict],
        include_system: bool = True
    ) -> str:
        """
        Build complete prompt with optional system message.
        
        For models that support system messages (like Llama 3.2), this formats
        as separate system/user messages. Otherwise combines them.
        """
        user_prompt = self.build_user_prompt(query, extracted_contexts)
        
        if include_system:
            return f"{self.system_prompt}\n\n{user_prompt}"
        
        return user_prompt
    
    def format_for_chat_model(
        self,
        query: str,
        extracted_contexts: List[Dict]
    ) -> List[Dict[str, str]]:
        """
        Format as chat messages for models expecting [{"role": ..., "content": ...}]
        
        Returns:
            List of message dicts for chat-based models
        """
        return [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": self.build_user_prompt(query, extracted_contexts)
            }
        ]


def main():
    """Demo: Show prompt formatting."""
    
    # Sample extracted contexts
    sample_contexts = [
        {
            'arxiv_id': '2602.05298v1',
            'section': 'introduction',
            'relevance': 0.64,
            'extracted_context': 'A key empirical finding in language models is the emergence of scaling laws...'
        },
        {
            'arxiv_id': '2602.04998v1',
            'section': 'related work',
            'relevance': 0.61,
            'extracted_context': 'Learning rate selection is crucial for optimization. Too large causes instability...'
        }
    ]
    
    query = "How to choose the optimal learning rate?"
    
    builder = PromptBuilder()
    
    print("=" * 80)
    print("CHAT FORMAT (for Llama 3.2):")
    print("=" * 80)
    messages = builder.format_for_chat_model(query, sample_contexts)
    for msg in messages:
        print(f"\n[{msg['role'].upper()}]")
        print(msg['content'][:300] + "..." if len(msg['content']) > 300 else msg['content'])
    
    print("\n\n" + "=" * 80)
    print("SINGLE STRING FORMAT (alternative):")
    print("=" * 80)
    full_prompt = builder.build_full_prompt(query, sample_contexts)
    print(full_prompt[:500] + "..." if len(full_prompt) > 500 else full_prompt)


if __name__ == "__main__":
    main()