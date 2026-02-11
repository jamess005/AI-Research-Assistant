"""
Improved LaTeX Parser with Aggressive Mathematical Content Removal
Fixes issues with:
- Mathematical symbols leaking through
- Citation markers (<cit.>, <ref>)
- Incomplete sentences
- Section symbols
- Unicode artifacts
"""

import tarfile
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Optional
from pylatexenc.latex2text import LatexNodes2Text
import json


class LatexParser:
    """Parse LaTeX source files with aggressive cleaning."""
    
    SECTIONS = [
        'abstract',
        'introduction',
        'related work',
        'background',
        'methodology',
        'method',
        'approach',
        'model',
        'experiments',
        'results',
        'discussion',
        'conclusion',
    ]
    
    SKIP_SECTIONS = [
        'references',
        'bibliography',
        'acknowledgments',
        'acknowledgements',
        'appendix',
        'supplementary',
        'funding',
        'author contributions',
    ]
    
    def __init__(self):
        """Initialize the improved LaTeX parser."""
        # Configure pylatexenc to be more robust and handle edge cases better
        self.converter = LatexNodes2Text(
            math_mode='remove',  # Remove math entirely to avoid parsing issues
            strict_latex_spaces=False,  # Be flexible with whitespace
            keep_braced_groups=False,  # Don't preserve braces
            keep_comments=False  # Strip comments
        )
        self.fallback_count = 0  # Track how many times we use fallback
    
    def extract_from_tarball(self, tar_path: str) -> Optional[str]:
        """Extract LaTeX source from ArXiv tarball."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Try to open as gzip tarball
                try:
                    with tarfile.open(tar_path, 'r:gz') as tar:
                        tar.extractall(tmpdir)
                except (tarfile.ReadError, OSError) as e:
                    # Corrupted file - delete it so it can be re-downloaded
                    print(f"Error: Corrupted tarball - {e}")
                    Path(tar_path).unlink(missing_ok=True)
                    return None
                
                tex_files = list(Path(tmpdir).rglob("*.tex"))
                
                if not tex_files:
                    print("Error: No .tex files found in tarball")
                    Path(tar_path).unlink(missing_ok=True)
                    return None
                
                # Filter out tiny files (likely not main document)
                tex_files = [f for f in tex_files if f.stat().st_size > 100]
                
                if not tex_files:
                    print("Error: No substantial .tex files found")
                    Path(tar_path).unlink(missing_ok=True)
                    return None
                
                # Strategy 1: Find the file with \begin{document} (that's the main file)
                main_tex = None
                for tf in tex_files:
                    try:
                        content = tf.read_text(encoding='utf-8', errors='ignore')
                        if r'\begin{document}' in content:
                            # Among files with \begin{document}, pick the largest
                            if main_tex is None or tf.stat().st_size > main_tex.stat().st_size:
                                main_tex = tf
                    except Exception:
                        continue
                
                # If main file found, resolve \input{} directives to inline sub-files
                if main_tex is not None:
                    latex_source = main_tex.read_text(encoding='utf-8', errors='ignore')
                    latex_source = self._resolve_inputs(latex_source, main_tex.parent, tmpdir)
                    if len(latex_source.strip()) >= 500:
                        return latex_source
                
                # Strategy 2: No \begin{document} found — multi-file project
                # Concatenate all files that contain \section content
                section_files = []
                for tf in sorted(tex_files, key=lambda f: f.name):
                    try:
                        content = tf.read_text(encoding='utf-8', errors='ignore')
                        if re.search(r'\\section\b', content):
                            section_files.append(tf)
                    except Exception:
                        continue
                
                if section_files:
                    combined = []
                    for sf in section_files:
                        combined.append(sf.read_text(encoding='utf-8', errors='ignore'))
                    latex_source = '\n\n'.join(combined)
                    if len(latex_source.strip()) >= 500:
                        return latex_source
                
                # Strategy 3: Fall back to largest .tex file
                main_tex = max(tex_files, key=lambda f: f.stat().st_size)
                
                try:
                    with open(main_tex, 'r', encoding='utf-8', errors='ignore') as f:
                        latex_source = f.read()
                    
                    # Validate content
                    if len(latex_source.strip()) < 500:
                        print("Error: LaTeX file too short")
                        Path(tar_path).unlink(missing_ok=True)
                        return None
                    
                    return latex_source
                except Exception as e:
                    print(f"Error reading tex file: {e}")
                    return None
                
        except Exception as e:
            print(f"Error extracting tarball: {e}")
            # Delete problematic file
            try:
                Path(tar_path).unlink(missing_ok=True)
            except Exception:
                pass
            return None
    
    def _resolve_inputs(self, latex_source: str, base_dir: Path, root_dir: str) -> str:
        """
        Resolve \\input{} and \\include{} directives by inlining referenced files.
        Handles up to 3 levels of nesting.
        """
        for _ in range(3):  # max nesting depth
            def _replace_input(match):
                filename = match.group(1)
                # Try with and without .tex extension
                candidates = [
                    base_dir / filename,
                    base_dir / f"{filename}.tex",
                    Path(root_dir) / filename,
                    Path(root_dir) / f"{filename}.tex",
                ]
                for candidate in candidates:
                    if candidate.exists() and candidate.is_file():
                        try:
                            return candidate.read_text(encoding='utf-8', errors='ignore')
                        except Exception:
                            pass
                return ''  # File not found, remove the directive
            
            new_source = re.sub(r'\\(?:input|include)\{([^}]+)\}', _replace_input, latex_source)
            if new_source == latex_source:
                break
            latex_source = new_source
        
        return latex_source
    
    def convert_latex_to_text(self, latex_source: str) -> str:
        """
        Convert LaTeX to plain text with aggressive math removal.
        """
        # Strip preamble FIRST — everything before \begin{document}
        # This prevents pylatexenc from choking on \newcommand, \usepackage, etc.
        doc_start = re.search(r'\\begin\{document\}', latex_source)
        if doc_start:
            latex_source = latex_source[doc_start.end():]
        # Strip \end{document} and everything after
        latex_source = re.sub(r'\\end\{document\}.*', '', latex_source, flags=re.DOTALL)
        
        # Remove comments
        latex_source = re.sub(r'%.*', '', latex_source)
        
        # Remove ALL math environments (display)
        latex_source = re.sub(r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}', '', latex_source, flags=re.DOTALL)
        latex_source = re.sub(r'\\begin\{align\*?\}.*?\\end\{align\*?\}', '', latex_source, flags=re.DOTALL)
        latex_source = re.sub(r'\\begin\{gather\*?\}.*?\\end\{gather\*?\}', '', latex_source, flags=re.DOTALL)
        latex_source = re.sub(r'\\begin\{eqnarray\*?\}.*?\\end\{eqnarray\*?\}', '', latex_source, flags=re.DOTALL)
        latex_source = re.sub(r'\\begin\{split\}.*?\\end\{split\}', '', latex_source, flags=re.DOTALL)
        latex_source = re.sub(r'\\begin\{multline\*?\}.*?\\end\{multline\*?\}', '', latex_source, flags=re.DOTALL)
        latex_source = re.sub(r'\\\[.*?\\\]', '', latex_source, flags=re.DOTALL)
        latex_source = re.sub(r'\$\$.*?\$\$', '', latex_source, flags=re.DOTALL)
        
        # Remove inline math (be aggressive)
        latex_source = re.sub(r'\$[^\$]+\$', ' ', latex_source)
        latex_source = re.sub(r'\\\(.*?\\\)', ' ', latex_source)
        
        # Remove figure and table environments
        latex_source = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', '', latex_source, flags=re.DOTALL)
        latex_source = re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', '', latex_source, flags=re.DOTALL)
        
        # Remove bibliography
        latex_source = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', '', latex_source, flags=re.DOTALL)
        
        # Convert to text using pylatexenc
        used_fallback = False
        try:
            text = self.converter.latex_to_text(latex_source)
        except Exception:
            # pylatexenc occasionally fails on complex LaTeX constructs
            # Silently use our robust fallback parser instead
            used_fallback = True
            self.fallback_count += 1
            text = self._fallback_latex_to_text(latex_source)
        
        # Post-processing cleanup (less aggressive for fallback text)
        text = self._aggressive_math_cleanup(text, preserve_content=used_fallback)
        
        return text
    
    def _fallback_latex_to_text(self, latex_source: str) -> str:
        """
        Robust fallback LaTeX→text converter for when pylatexenc fails.
        Handles nested braces properly by iterating.
        """
        text = latex_source
        
        # Strip preamble: everything before \begin{document}
        doc_match = re.search(r'\\begin\{document\}', text)
        if doc_match:
            text = text[doc_match.end():]
        # Strip \end{document}
        text = re.sub(r'\\end\{document\}.*', '', text, flags=re.DOTALL)
        
        # Remove comments
        text = re.sub(r'%.*', '', text)
        
        # Remove all math environments
        for env in ['equation', 'align', 'gather', 'eqnarray', 'split', 'multline',
                     'figure', 'table', 'thebibliography', 'tikzpicture', 'lstlisting']:
            text = re.sub(rf'\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}', '', text, flags=re.DOTALL)
        
        # Remove display math
        text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
        text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
        text = re.sub(r'\$[^\$]+\$', ' ', text)
        
        # Iteratively resolve \command{content} → content (handles nesting)
        # Repeat until no more matches (peels one layer per pass)
        for _ in range(10):  # max 10 nesting levels
            new_text = re.sub(r'\\[a-zA-Z]+\{([^{}]*)\}', r'\1', text)
            if new_text == text:
                break
            text = new_text
        
        # Remove standalone commands (\command without braces)
        text = re.sub(r'\\[a-zA-Z]+\b', '', text)
        
        # Remove remaining braces
        text = text.replace('{', '').replace('}', '')
        
        # Remove backslashes
        text = re.sub(r'\\(?!n)', '', text)
        
        return text
    
    def _aggressive_math_cleanup(self, text: str, preserve_content: bool = False) -> str:
        """
        Remove mathematical content and LaTeX artifacts that leaked through.
        When preserve_content=True (fallback mode), keep text inside commands.
        """
        if preserve_content:
            # Fallback mode: extract content from any remaining \cmd{content}
            for _ in range(5):
                new_text = re.sub(r'\\[a-zA-Z]+\{([^{}]*)\}', r'\1', text)
                if new_text == text:
                    break
                text = new_text
            text = re.sub(r'\\[a-zA-Z]+\[[^\]]*\]', '', text)  # \cmd[opts]
            text = re.sub(r'\\[a-zA-Z]+\b', '', text)  # standalone commands
        else:
            # pylatexenc succeeded: no \commands should remain, just clean artifacts
            text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
            text = re.sub(r'\\[a-zA-Z]+\[[^\]]*\]', '', text)
            text = re.sub(r'\\[a-zA-Z]+\b', '', text)
        
        # Remove citation markers
        text = re.sub(r'<cit\.>', '', text)
        text = re.sub(r'<ref>', '', text)
        text = re.sub(r'\\cite\{[^}]*\}', '', text)
        text = re.sub(r'\\ref\{[^}]*\}', '', text)
        
        # Remove section symbols
        text = re.sub(r'§', '', text)
        text = text.replace('§', '')
        
        # Remove mathematical notation patterns
        # Variables with subscripts/superscripts: h_model, N_b, etc.
        text = re.sub(r'\b[a-zA-Z]_[a-zA-Z0-9]+\b', '', text)
        text = re.sub(r'\b[a-zA-Z]\^[a-zA-Z0-9]+\b', '', text)
        
        # Greek letters (unicode)
        greek_pattern = r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]'
        text = re.sub(greek_pattern, '', text)
        
        # Mathematical symbols (unicode)
        math_symbols = r'[∑∫∂∇∆∏√∞≈≠≤≥±×÷∈∉⊂⊃∪∩⊆⊇∧∨¬⟹⟺∀∃]'
        text = re.sub(math_symbols, '', text)
        
        # Remove sequences of mathematical operations
        # Pattern: things like "e^x^2" or "i≥j'" 
        text = re.sub(r'[a-zA-Z0-9]+[\^_≥≤=<>]+[a-zA-Z0-9\^_≥≤=<>\']*', '', text)
        
        # Remove fragments like "where denotes" with no context
        # These happen when equations get removed
        text = re.sub(r'\bwhere\s+denotes\s*', '', text)
        text = re.sub(r'\bwhere\s+is\s+the\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\bdenotes\s+the\s*$', '', text, flags=re.MULTILINE)
        
        # Remove incomplete metric descriptions like "50" at end of line
        text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove lines that are mostly mathematical garbage
        # If a line has more than 5 mathematical-looking characters, remove it
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Count math-like characters
            math_chars = len(re.findall(r'[_\^≥≤≈∼θδσαβγ]', line))
            if math_chars < 5:  # Allow some, but not too many
                cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)
        
        # Clean up Unicode artifacts
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        text = text.replace('\u2014', '-')  # Em dash
        text = text.replace('\u2013', '-')  # En dash
        text = text.replace('\u2019', "'")  # Right single quote
        text = text.replace('\u201c', '"')  # Left double quote
        text = text.replace('\u201d', '"')  # Right double quote
        text = text.replace('\u2026', '...')  # Ellipsis
        
        # Remove escaped characters that leak through
        text = text.replace('\\"', '"')  # Escaped double quotes
        text = text.replace("\\'", "'")  # Escaped single quotes  
        text = text.replace('\\`', '`')  # Escaped backticks
        text = re.sub(r'\\(["\'`])', r'\1', text)  # Any other escaped quotes
        
        # Remove LaTeX compilation warnings and artifacts
        text = re.sub(r'get arXiv to do \d+ passes:.*?Rerun', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Label\(s\) may have changed\..*?', '', text)
        text = re.sub(r'\\\[.*?\\\]', '', text)  # Escaped brackets
        text = re.sub(r'\\\(.*?\\\)', '', text)  # Escaped parens
        
        # Remove standalone brackets/symbols artifacts
        text = re.sub(r'^\s*[\[\]{}]\s*$', '', text, flags=re.MULTILINE)
        
        # Clean up any remaining backslash artifacts (but preserve newlines)
        text = re.sub(r'\\(?!n)', '', text)  # Remove backslashes not followed by 'n'
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove excessive newlines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove empty lines that only have whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from paper text."""
        sections = {}
        
        section_positions = []
        
        for section in self.SECTIONS:
            # Try multiple capitalizations since pylatexenc outputs § INTRODUCTION (caps)
            found = False
            for section_var in [section.lower(), section.title(), section.upper(), section.capitalize()]:
                patterns = [
                    rf'(?:^|\n)\s*§\s*{re.escape(section_var)}\s*(?:\n|$)',  # § INTRODUCTION
                    rf'(?:^|\n)\s*\d+\.?\s+{re.escape(section_var)}\s*(?:\n|$)',  # 1. Introduction
                    rf'(?:^|\n)\s*{re.escape(section_var)}\s*(?:\n|$)',  # Introduction
                    rf'(?:^|\n)\s*\d+\.\d+\.?\s+{re.escape(section_var)}\s*(?:\n|$)',  # 1.1 Introduction
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text, re.MULTILINE)
                    if match:
                        section_positions.append((match.start(), section))
                        found = True
                        break
                
                if found:
                    break
        
        section_positions.sort(key=lambda x: x[0])
        
        for i, (start_pos, section_name) in enumerate(section_positions):
            if i < len(section_positions) - 1:
                end_pos = section_positions[i + 1][0]
            else:
                end_pos = len(text)
                # Check for skip sections with multiple capitalizations
                for skip_section in self.SKIP_SECTIONS:
                    for skip_var in [skip_section.lower(), skip_section.title(), skip_section.upper()]:
                        pattern = rf'(?:^|\n)\s*{re.escape(skip_var)}\s*(?:\n|$)'
                        match = re.search(pattern, text[start_pos:], re.MULTILINE)
                        if match:
                            end_pos = start_pos + match.start()
                            break
                    if end_pos != len(text):
                        break
            
            section_text = text[start_pos:end_pos]
            section_text = self._clean_section_text(section_text)
            
            if section_text and len(section_text) > 100:
                sections[section_name] = section_text
        
        return sections
    
    def _clean_section_text(self, text: str) -> str:
        """Clean section text with enhanced formatting cleanup."""
        lines = text.split('\n')
        if lines:
            lines = lines[1:]  # Skip section header
        
        text = '\n'.join(lines)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove excessive newlines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove very short lines (likely headers/footers)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line.strip()) > 20 or line.strip() == '']
        text = '\n'.join(cleaned_lines)
        
        # Final pass: normalize whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text
    
    def parse_paper(self, 
                    tar_path: str,
                    arxiv_id: str,
                    metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Parse a complete paper from tarball."""
        latex_source = self.extract_from_tarball(tar_path)
        
        if not latex_source:
            return None
        
        text = self.convert_latex_to_text(latex_source)
        
        if not text or len(text) < 500:
            return None
        
        sections = self.extract_sections(text)
        
        if not sections:
            sections = {'full_text': text}
        
        paper_data = {
            'arxiv_id': arxiv_id,
            'parsing_method': 'latex',
            'sections': sections,
            'full_text': text,
            'text_length': len(text),
            'word_count': len(text.split()),
            'section_count': len(sections),
        }
        
        if metadata:
            paper_data['title'] = metadata.get('title', '')
            paper_data['authors'] = metadata.get('authors', [])
            paper_data['categories'] = metadata.get('categories', [])
            paper_data['published'] = metadata.get('published', '')
        
        return paper_data
    
    def parse_directory(self,
                       input_dir: str,
                       output_dir: str,
                       metadata_file: Optional[str] = None) -> int:
        """Parse all tarballs in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metadata_dict = {}
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                metadata_list = json.load(f)
                for item in metadata_list:
                    arxiv_id = item.get('arxiv_id', '')
                    metadata_dict[arxiv_id] = item
        
        tar_files = list(input_path.glob("*.tar.gz"))
        print(f"Found {len(tar_files)} LaTeX source tarballs to parse")
        
        successful = 0
        
        for tar_file in tar_files:
            print(f"Parsing {tar_file.name}...")
            
            arxiv_id = tar_file.stem.replace('.tar', '')
            
            paper_data = self.parse_paper(
                str(tar_file),
                arxiv_id,
                metadata_dict.get(arxiv_id)
            )
            
            if not paper_data:
                print(f"  Failed to parse {tar_file.name}")
                continue
            
            output_file = output_path / f"{arxiv_id}.json"
            with open(output_file, 'w') as f:
                json.dump(paper_data, f, indent=2)
            
            successful += 1
            print(f"  Extracted {paper_data['word_count']} words, {paper_data['section_count']} sections")
        
        print(f"\nSuccessfully parsed {successful}/{len(tar_files)} papers")
        return successful