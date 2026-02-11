"""
Improved ArXiv Downloader
Downloads LaTeX source (.tar.gz) only
Tracks download status and parsing method availability
"""

import os
import time
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Any
import yaml
from tqdm import tqdm


class ArxivClient:
    """Client for downloading ArXiv papers with LaTeX priority."""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    LATEX_URL = "https://arxiv.org/e-print"  # LaTeX source endpoint
    PDF_URL = "https://arxiv.org/pdf"        # PDF endpoint
    
    def __init__(self, config_path: str = "config/arxiv_config.yaml"):
        """Initialize the ArXiv client."""
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = Path.cwd() / self.config_path
        
        self.project_root = self.config_path.parent.parent
        self.config = self._load_config(str(self.config_path))
        
        # Set up directories
        pdf_dir_relative = self.config['download']['pdf_dir']
        self.download_dir = self.project_root / pdf_dir_relative
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        metadata_file_relative = self.config['metadata']['metadata_file']
        self.metadata_file = self.project_root / metadata_file_relative
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.metadata = []
        
        # Set up requests session with proper headers and retry logic
        self.session = requests.Session()
        
        # ArXiv requires User-Agent header
        self.session.headers.update({
            'User-Agent': 'AIResearchRAG/1.0 (research project; mailto:user@example.com)'
        })
        
        # Configure retry strategy for rate limiting
        retry_strategy = Retry(
            total=3,  # Reduced from 5 - let manual retry handle persistent issues
            backoff_factor=5,  # Increased from 3 - wait 5, 10, 20 seconds
            status_forcelist=[500, 502, 503, 504],  # Removed 429 - handle manually
            allowed_methods=["GET", "HEAD"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _build_search_query(self) -> str:
        """Build the ArXiv API search query string."""
        categories = self.config['search']['categories']
        query = self.config['search'].get('query', '')
        
        cat_queries = [f"cat:{cat}" for cat in categories]
        category_query = " OR ".join(cat_queries)
        
        if query:
            full_query = f"({category_query}) AND ({query})"
        else:
            full_query = category_query
        
        return full_query
    
    def search_papers(self, max_results: Optional[int] = None, start: int = 0) -> List[Dict]:
        """
        Search for papers on ArXiv.
        
        Args:
            max_results: Maximum number of results
            start: Offset into search results (for pagination).
                   Use this to skip papers already downloaded on prior runs.
            
        Returns:
            List of paper metadata dictionaries
        """
        if max_results is None:
            max_results = self.config['search']['max_results']
        
        search_query = self._build_search_query()
        sort_by = self.config['search']['sort_by']
        sort_order = self.config['search']['sort_order']
        
        params = {
            'search_query': search_query,
            'start': start,
            'max_results': max_results,
            'sortBy': sort_by,
            'sortOrder': sort_order
        }
        
        print(f"Searching ArXiv for: {search_query}")
        print(f"Max results: {max_results}, offset: {start}")
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error searching ArXiv: {e}")
            if "429" in str(e):
                print("\nRate limit exceeded. Waiting 10 seconds before retry...")
                time.sleep(10)
                try:
                    response = self.session.get(self.BASE_URL, params=params, timeout=30)
                    response.raise_for_status()
                except requests.RequestException as e2:
                    print(f"Retry failed: {e2}")
                    print("\nTip: ArXiv rate limits are strict. Try:")
                    print("   - Reducing number of papers to download")
                    print("   - Waiting a few minutes before retrying")
                    print("   - Using smaller batch sizes")
                    return []
            else:
                return []
        
        # Parse XML response
        papers = self._parse_search_response(response.text)
        print(f"Found {len(papers)} papers")
        
        self.metadata = papers
        return papers
    
    def _parse_search_response(self, xml_text: str) -> List[Dict]:
        """Parse ArXiv API XML response."""
        papers = []
        root = ET.fromstring(xml_text)
        
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        for entry in root.findall('atom:entry', ns):
            try:
                id_elem = entry.find('atom:id', ns)
                title_elem = entry.find('atom:title', ns)
                summary_elem = entry.find('atom:summary', ns)
                published_elem = entry.find('atom:published', ns)
                updated_elem = entry.find('atom:updated', ns)
                
                if not all([id_elem is not None and id_elem.text,
                           title_elem is not None and title_elem.text,
                           summary_elem is not None and summary_elem.text,
                           published_elem is not None and published_elem.text,
                           updated_elem is not None and updated_elem.text]):
                    continue
                
                assert id_elem is not None and id_elem.text is not None
                assert title_elem is not None and title_elem.text is not None
                assert summary_elem is not None and summary_elem.text is not None
                assert published_elem is not None and published_elem.text is not None
                assert updated_elem is not None and updated_elem.text is not None
                
                paper: Dict[str, Any] = {
                    'id': id_elem.text,
                    'title': title_elem.text.strip().replace('\n', ' '),
                    'summary': summary_elem.text.strip().replace('\n', ' '),
                    'published': published_elem.text,
                    'updated': updated_elem.text,
                }
                
                # Extract authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns)
                    if name is not None:
                        authors.append(name.text)
                paper['authors'] = authors
                
                # Extract categories
                categories = []
                for category in entry.findall('atom:category', ns):
                    term = category.get('term')
                    if term:
                        categories.append(term)
                paper['categories'] = categories
                
                # Extract ArXiv ID
                arxiv_id = paper['id'].split('/abs/')[-1]
                paper['arxiv_id'] = arxiv_id
                
                # Build download URLs
                paper['latex_url'] = f"{self.LATEX_URL}/{arxiv_id}"
                paper['pdf_url'] = f"{self.PDF_URL}/{arxiv_id}.pdf"
                
                papers.append(paper)
                
            except Exception as e:
                print(f"Error parsing entry: {e}")
                continue
        
        return papers
    
    def check_latex_availability(self, arxiv_id: str) -> bool:
        """
        Check if LaTeX source is available for a paper.
        
        Args:
            arxiv_id: ArXiv ID
            
        Returns:
            True if LaTeX source is available
        """
        url = f"{self.LATEX_URL}/{arxiv_id}"
        try:
            response = self.session.head(url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def download_paper(self, paper: Dict) -> Dict[str, Any]:
        """
        Download a single paper (LaTeX source only).
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            Dictionary with download status
        """
        arxiv_id = paper['arxiv_id']
        request_delay = self.config['download']['request_delay']
        timeout = self.config['download']['timeout']
        
        result = {
            'arxiv_id': arxiv_id,
            'latex_available': False,
            'downloaded': False,
            'download_method': None,
            'file_path': None,
        }
        
        # Try LaTeX source first
        latex_path = self.download_dir / f"{arxiv_id}.tar.gz"
        if not latex_path.exists():
            try:
                time.sleep(request_delay)  # Rate limit
                response = self.session.get(
                    paper['latex_url'],
                    timeout=timeout,
                    stream=True
                )
                
                if response.status_code == 200:
                    with open(latex_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Validate file isn't empty
                    if latex_path.stat().st_size == 0:
                        latex_path.unlink(missing_ok=True)
                        print(f"  LaTeX download was empty for {arxiv_id}")
                    else:
                        result['latex_available'] = True
                        result['downloaded'] = True
                        result['download_method'] = 'latex'
                        result['file_path'] = str(latex_path)
                        print(f"  Downloaded LaTeX source: {arxiv_id}.tar.gz")
                        time.sleep(request_delay)
                        return result
                    
            except Exception as e:
                print(f"  LaTeX download failed for {arxiv_id}: {e}")
        else:
            result['latex_available'] = True
            result['downloaded'] = True
            result['download_method'] = 'latex'
            result['file_path'] = str(latex_path)
            return result
        
        # No PDF fallback - treat as failed
        print(f"  LaTeX unavailable for {arxiv_id} (skipping, no PDF fallback)")
        
        time.sleep(request_delay)
        return result
    
    def download_papers(self,
                       papers: Optional[List[Dict]] = None,
                       start_idx: int = 0) -> Dict[str, int]:
        """
        Download papers with LaTeX priority.
        
        Args:
            papers: List of paper metadata
            start_idx: Index to start from
            
        Returns:
            Statistics dictionary
        """
        if papers is None:
            papers = self.metadata
        
        if not papers:
            print("No papers to download")
            return {'total': 0, 'latex': 0, 'pdf': 0, 'failed': 0}
        
        print(f"\nDownloading {len(papers[start_idx:])} papers...")
        print("LaTeX source only (no PDF fallback)")
        
        stats = {
            'total': len(papers[start_idx:]),
            'latex': 0,
            'pdf': 0,
            'failed': 0,
            'skipped': 0,  # New: track already-existing files
        }
        
        download_results = []
        
        for paper in tqdm(papers[start_idx:], initial=start_idx):
            arxiv_id = paper['arxiv_id']
            
            # Check if file already exists BEFORE attempting download
            latex_exists = (self.download_dir / f"{arxiv_id}.tar.gz").exists()
            
            if latex_exists:
                stats['skipped'] += 1
                result = {
                    'arxiv_id': arxiv_id,
                    'latex_available': True,
                    'downloaded': False,
                    'download_method': 'latex',
                    'file_path': str(self.download_dir / f"{arxiv_id}.tar.gz"),
                }
                download_results.append(result)
                continue
            
            # Actually download if not exists
            result = self.download_paper(paper)
            download_results.append(result)
            
            if result['downloaded']:
                if result['download_method'] == 'latex':
                    stats['latex'] += 1
                elif result['download_method'] == 'pdf':
                    stats['pdf'] += 1
            else:
                stats['failed'] += 1
        
        # Save enhanced metadata with download info
        for paper, result in zip(papers[start_idx:], download_results):
            paper['latex_available'] = result['latex_available']
            paper['download_method'] = result['download_method']
        
        self._save_metadata(papers_to_save=papers[start_idx:])
        
        print(f"\n{'='*80}")
        print("Download Summary")
        print(f"{'='*80}")
        print(f"  Total papers: {stats['total']}")
        print(f"  LaTeX source: {stats['latex']} ({stats['latex']/stats['total']*100:.1f}%)")
        if stats['skipped'] > 0:
            print(f"  Already downloaded: {stats['skipped']} ({stats['skipped']/stats['total']*100:.1f}%)")
        print(f"  Failed (no LaTeX): {stats['failed']}")
        
        return stats
    
    def _save_metadata(self, papers_to_save: Optional[List[Dict]] = None):
        """Save metadata, merging with existing file to preserve history."""
        save_list = papers_to_save if papers_to_save is not None else self.metadata
        
        # Load existing metadata and merge
        existing = []
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
            except (json.JSONDecodeError, Exception):
                existing = []
        
        # Build lookup of existing IDs
        existing_ids = {p.get('arxiv_id', ''): i for i, p in enumerate(existing)}
        
        # Add or update entries
        for paper in save_list:
            pid = paper.get('arxiv_id', '')
            if pid in existing_ids:
                existing[existing_ids[pid]] = paper
            else:
                existing.append(paper)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"\nMetadata saved to {self.metadata_file} ({len(existing)} papers)")
    
    def load_metadata(self) -> List[Dict]:
        """Load metadata from JSON file."""
        if not self.metadata_file.exists():
            return []
        
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        return self.metadata


if __name__ == "__main__":
    # Test the downloader
    client = ArxivClient()
    
    # Search for papers
    papers = client.search_papers(max_results=5)
    
    # Download
    if papers:
        stats = client.download_papers(papers)
        print(f"\nDownloaded {stats['latex']} LaTeX sources, {stats['pdf']} PDFs")