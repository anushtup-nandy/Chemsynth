# utils/scifinder_processor.py
import requests
import time
from typing import List, Dict, Optional, Any
from datetime import datetime

class SciFinderProcessor:
    """
    Processor for interacting with the CAS SciFinder API to search for scientific literature.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://commonchemistry.cas.org/api"
        self.headers = {
            "X-CAS-TOKEN": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
    def search_by_cas_number(self, cas_number: str) -> Optional[Dict[str, Any]]:
        """
        Search for a compound by CAS Registry Number.
        
        Args:
            cas_number: CAS Registry Number (e.g., "50-00-0")
            
        Returns:
            Dictionary containing compound information or None if not found
        """
        try:
            url = f"{self.base_url}/detail"
            params = {"cas_rn": cas_number}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data if data else None
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching CAS number {cas_number}: {e}")
            return None
    
    def search_by_name(self, compound_name: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for compounds by name.
        
        Args:
            compound_name: Name of the compound to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of compound dictionaries
        """
        try:
            url = f"{self.base_url}/search"
            payload = {
                "q": compound_name,
                "size": min(max_results, 50)  # API limit
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data.get("results", []) if data else []
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching compound name '{compound_name}': {e}")
            return []
    
    def search_by_smiles(self, smiles: str) -> Optional[Dict[str, Any]]:
        """
        Search for a compound by SMILES string.
        Note: This may require the structure search endpoint which might not be available in Common Chemistry API.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Dictionary containing compound information or None if not found
        """
        try:
            # For Common Chemistry API, we might need to use a different approach
            # as direct SMILES search may not be available
            url = f"{self.base_url}/search"
            payload = {
                "q": smiles,  # Try searching SMILES as text
                "size": 5
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", []) if data else []
            
            # Return the first result if any found
            return results[0] if results else None
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching SMILES '{smiles}': {e}")
            return None
    
    def get_references_for_compound(self, cas_number: str = None, compound_data: Dict = None) -> List[Dict[str, Any]]:
        """
        Extract literature references for a compound.
        
        Args:
            cas_number: CAS Registry Number
            compound_data: Compound data dictionary (if already retrieved)
            
        Returns:
            List of reference dictionaries formatted for literature analysis
        """
        references = []
        
        try:
            # If compound data not provided, fetch it
            if not compound_data and cas_number:
                compound_data = self.search_by_cas_number(cas_number)
            
            if not compound_data:
                return references
            
            # Extract references from compound data
            # The exact structure depends on the API response format
            refs = compound_data.get("references", [])
            
            for ref in refs[:10]:  # Limit to 10 references
                # Format reference data to match the expected structure
                formatted_ref = {
                    "title": ref.get("title", "No title available"),
                    "authors": self._extract_authors(ref.get("authors", [])),
                    "published": self._extract_date(ref.get("publicationDate", "")),
                    "source_db": "SciFinder (CAS)",
                    "summary": ref.get("abstract", ref.get("summary", "No abstract available")),
                    "doi": ref.get("doi", ""),
                    "pdf_url": ref.get("url", ""),
                    "entry_id": ref.get("id", f"cas_{int(time.time())}")
                }
                references.append(formatted_ref)
                
        except Exception as e:
            print(f"Error extracting references: {e}")
            
        return references
    
    def _extract_authors(self, authors_data: List[Any]) -> List[str]:
        """Extract and format author names from API response."""
        authors = []
        try:
            for author in authors_data[:5]:  # Limit to 5 authors
                if isinstance(author, str):
                    authors.append(author)
                elif isinstance(author, dict):
                    name = author.get("name") or f"{author.get('firstName', '')} {author.get('lastName', '')}".strip()
                    if name:
                        authors.append(name)
        except:
            pass
        return authors
    
    def _extract_date(self, date_str: str) -> str:
        """Extract and format publication date."""
        try:
            if date_str:
                # Try to parse different date formats
                for fmt in ["%Y-%m-%d", "%Y-%m", "%Y", "%d/%m/%Y", "%m/%d/%Y"]:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        return date_obj.strftime("%Y-%m-%d")
                    except ValueError:
                        continue
                return date_str  # Return as-is if parsing fails
        except:
            pass
        return "Unknown date"


def search_scifinder_literature(query: str, max_results: int = 5, api_key: str = "GltnljfNwM7IUX8X0B7H64ptrqepLzbR7RmX7kfN") -> List[Dict[str, Any]]:
    """
    Search for literature using SciFinder API.
    
    Args:
        query: Search query (compound name, CAS number, or SMILES)
        max_results: Maximum number of results to return
        api_key: CAS API key
        
    Returns:
        List of formatted paper dictionaries
    """
    processor = SciFinderProcessor(api_key)
    papers = []
    
    try:
        print(f"Searching SciFinder for: '{query}'")
        
        # Determine query type and search accordingly
        if _is_cas_number(query):
            # Search by CAS number
            compound_data = processor.search_by_cas_number(query)
            if compound_data:
                papers = processor.get_references_for_compound(compound_data=compound_data)
                
        elif _is_smiles(query):
            # Search by SMILES
            compound_data = processor.search_by_smiles(query)
            if compound_data:
                papers = processor.get_references_for_compound(compound_data=compound_data)
                
        else:
            # Search by compound name
            compounds = processor.search_by_name(query, max_results=3)
            
            # For each found compound, get its references
            for compound in compounds[:2]:  # Limit to first 2 compounds
                cas_number = compound.get("rn")  # Registry number
                if cas_number:
                    refs = processor.get_references_for_compound(cas_number=cas_number)
                    papers.extend(refs)
                    
                    if len(papers) >= max_results:
                        break
        
        # Deduplicate by DOI or title
        seen = set()
        unique_papers = []
        for paper in papers:
            identifier = paper.get("doi") or paper.get("title", "")
            if identifier and identifier not in seen:
                seen.add(identifier)
                unique_papers.append(paper)
                
                if len(unique_papers) >= max_results:
                    break
        
        print(f"SciFinder search returned {len(unique_papers)} unique papers")
        return unique_papers
        
    except Exception as e:
        print(f"Error in SciFinder literature search: {e}")
        return []


def _is_cas_number(query: str) -> bool:
    """Check if query looks like a CAS Registry Number."""
    import re
    cas_pattern = r'^\d{2,7}-\d{2}-\d$'
    return bool(re.match(cas_pattern, query.strip()))


def _is_smiles(query: str) -> bool:
    """Check if query looks like a SMILES string."""
    import re
    # Basic SMILES pattern - contains common SMILES characters
    smiles_chars = set("CNOSPFClBrI[]()=-+#@0123456789")
    query_chars = set(query.replace(" ", ""))
    
    # If query is mostly SMILES characters and has some structure indicators
    if len(query_chars.intersection(smiles_chars)) / len(query_chars) > 0.7:
        return any(char in query for char in "()[]=-#")
    
    return False


if __name__ == "__main__":
    # Test the SciFinder processor
    test_queries = [
        "aspirin",
        "50-78-2",  # CAS number for aspirin
        "CC(=O)OC1=CC=CC=C1C(=O)O"  # SMILES for aspirin
    ]
    
    for query in test_queries:
        print(f"\n--- Testing SciFinder search for: '{query}' ---")
        results = search_scifinder_literature(query, max_results=3)
        
        if results:
            for i, paper in enumerate(results, 1):
                print(f"{i}. {paper['title']}")
                print(f"   Authors: {', '.join(paper['authors'][:3])}")
                print(f"   Published: {paper['published']}")
                print(f"   Abstract: {paper['summary'][:150]}...")
                print()
        else:
            print("No results found or error occurred.")