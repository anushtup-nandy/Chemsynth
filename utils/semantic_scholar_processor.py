"""
Fixed Semantic Scholar API processor for literature search
Addresses the zero results issue with proper endpoint usage and fallback strategies
"""

import requests
import time
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
import logging
import json

class SemanticScholarProcessor:
    """
    Processor for Semantic Scholar Academic Graph API
    """
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 1.1):
        """
        Initialize the Semantic Scholar processor
        
        Args:
            api_key: Optional API key for higher rate limits
            rate_limit_delay: Delay between requests in seconds (1.1s for safety with 1req/s limit)
        """
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        
        # Set headers
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})
        
        self.session.headers.update({
            "User-Agent": "ChemicalSynthesisAnalyzer/1.0",
            "Accept": "application/json"
        })
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make a rate-limited request to Semantic Scholar API
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response data or None if failed
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                self.logger.warning("Rate limit hit, waiting longer...")
                time.sleep(5)  # Wait longer on rate limit
                return self._make_request(endpoint, params)  # Retry once
            elif response.status_code == 400:
                self.logger.error(f"Bad request to Semantic Scholar: {response.text}")
                return None
            else:
                self.logger.warning(f"Semantic Scholar API returned status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed to Semantic Scholar: {e}")
            return None
    
    def search_papers(self, 
                     query: str, 
                     compound_name: str = None,
                     max_results: int = 10,
                     fields: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for papers using Semantic Scholar's paper search endpoint
        
        Args:
            query: Original search query
            compound_name: Resolved compound name for better searching
            max_results: Maximum number of results to return
            fields: Specific fields to retrieve
            
        Returns:
            List of paper dictionaries
        """
        if fields is None:
            fields = [
                'paperId', 'title', 'abstract', 'year', 'authors', 'venue', 'url', 
                'citationCount', 'fieldsOfStudy', 'publicationDate', 's2FieldsOfStudy'
            ]
        
        # Try multiple search strategies
        search_terms = self._generate_search_terms(query, compound_name)
        all_papers = []
        
        for search_term in search_terms:
            self.logger.info(f"Searching Semantic Scholar with: '{search_term}'")
            
            # Use the simpler /paper/search endpoint instead of /paper/search/bulk
            params = {
                'query': search_term,
                'limit': min(max_results * 2, 100),  # Get more results to filter
                'fields': ','.join(fields),
                'year': '2010-',  # Focus on recent papers
            }
            
            # Try the regular search endpoint first
            response_data = self._make_request('/paper/search', params)
            
            if response_data and 'data' in response_data:
                papers = response_data['data']
                if papers:
                    self.logger.info(f"Found {len(papers)} papers for '{search_term}'")
                    all_papers.extend(papers)
                    break  # Stop on first successful search
            
            # If no results, try without year filter
            if not response_data or not response_data.get('data'):
                self.logger.info(f"No results with year filter, trying without...")
                params.pop('year', None)
                response_data = self._make_request('/paper/search', params)
                
                if response_data and 'data' in response_data:
                    papers = response_data['data']
                    if papers:
                        self.logger.info(f"Found {len(papers)} papers without year filter")
                        all_papers.extend(papers)
                        break
        
        # If still no results, try a very broad chemistry search as fallback
        if not all_papers:
            self.logger.info("No results found, trying broad chemistry search...")
            broad_terms = self._generate_fallback_terms(query, compound_name)
            
            for broad_term in broad_terms:
                params = {
                    'query': broad_term,
                    'limit': min(max_results, 20),
                    'fields': ','.join(fields)
                }
                
                response_data = self._make_request('/paper/search', params)
                if response_data and 'data' in response_data:
                    papers = response_data['data']
                    if papers:
                        self.logger.info(f"Found {len(papers)} papers with broad search '{broad_term}'")
                        all_papers.extend(papers)
                        break
        
        # Remove duplicates and process results
        unique_papers = self._deduplicate_papers(all_papers)
        processed_papers = []
        
        for paper in unique_papers:
            processed_paper = self._process_paper(paper, query)
            if processed_paper:
                processed_papers.append(processed_paper)
        
        # Sort by relevance and return top results
        processed_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        self.logger.info(f"Returning {len(processed_papers[:max_results])} processed papers")
        return processed_papers[:max_results]
    
    def _generate_search_terms(self, query: str, compound_name: str = None) -> List[str]:
        """
        Generate search terms in order of specificity
        """
        terms = []
        
        # Use compound name if available, otherwise original query
        primary_term = compound_name.strip() if compound_name else query.strip()
        
        # More targeted search terms for chemistry
        terms.extend([
            f'"{primary_term}" synthesis',
            f'"{primary_term}" preparation',
            f'"{primary_term}" chemical synthesis',
            f'{primary_term} synthesis method',
            primary_term  # Just the compound name
        ])
        
        # Add original query variations if different
        if compound_name and query.lower() != compound_name.lower():
            terms.extend([
                f'"{query}" synthesis',
                query
            ])
        
        return list(dict.fromkeys(terms))  # Remove duplicates while preserving order
    
    def _generate_fallback_terms(self, query: str, compound_name: str = None) -> List[str]:
        """
        Generate very broad fallback search terms
        """
        fallback_terms = []
        
        # Extract potential parent compound or key terms
        primary_term = compound_name or query
        
        # Try to extract meaningful chemical terms
        import re
        chemical_words = re.findall(r'[a-zA-Z]{4,}', primary_term.lower())
        
        for word in chemical_words[:3]:  # Use first 3 meaningful words
            if word not in ['acid', 'salt', 'chloro', 'bromo', 'methyl', 'ethyl']:
                fallback_terms.extend([
                    f'{word} synthesis',
                    f'{word} chemistry',
                    word
                ])
        
        # Generic chemistry searches as last resort
        fallback_terms.extend([
            'organic synthesis',
            'chemical synthesis methods',
            'synthetic chemistry'
        ])
        
        return list(dict.fromkeys(fallback_terms))
    
    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate papers based on paperId or title
        """
        seen = set()
        unique_papers = []
        
        for paper in papers:
            paper_id = paper.get('paperId')
            title = paper.get('title', '').lower().strip()
            
            identifier = paper_id or title
            if identifier and identifier not in seen:
                seen.add(identifier)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _process_paper(self, paper: Dict[str, Any], search_term: str) -> Optional[Dict[str, Any]]:
        """
        Process and score a paper for chemistry relevance
        """
        if not paper:
            return None
        
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        
        # Skip papers without title
        if not title:
            return None
        
        # Calculate relevance score
        relevance_score = self._calculate_chemistry_relevance(
            title.lower(), 
            abstract.lower() if abstract else '', 
            search_term
        )
        
        # Format authors
        authors = []
        for author in paper.get('authors', [])[:5]:  # Limit to first 5 authors
            if isinstance(author, dict) and 'name' in author:
                authors.append(author['name'])
        
        # Format venue and year information
        venue = paper.get('venue', '') or ''
        year = paper.get('year') or paper.get('publicationDate', '')[:4] if paper.get('publicationDate') else 'Unknown'
        
        # Format source information
        source_info = f"{', '.join(authors[:3])}{'...' if len(authors) > 3 else ''} ({year})"
        if venue:
            source_info += f" - {venue}"
        
        return {
            'title': title,
            'summary': abstract or 'Abstract not available',
            'authors': authors,
            'published': str(year),
            'source_db': 'Semantic Scholar',
            'pdf_url': paper.get('url'),
            'entry_id': paper.get('paperId'),
            'doi': paper.get('externalIds', {}).get('DOI') if paper.get('externalIds') else None,
            'citation_count': paper.get('citationCount', 0),
            'venue': venue,
            'relevance_score': relevance_score,
            'fields_of_study': [field.get('category') for field in paper.get('s2FieldsOfStudy', []) if field.get('category')],
            'source': source_info  # Formatted source for frontend
        }
    
    def _calculate_chemistry_relevance(self, title: str, abstract: str, search_term: str) -> int:
        """
        Calculate chemistry relevance score for a paper
        """
        score = 0
        search_lower = search_term.lower().replace('"', '').strip()
        text_content = f"{title} {abstract}"
        
        # Core chemistry keywords with weights
        chemistry_keywords = {
            'synthesis': 4, 'synthetic': 3, 'preparation': 3, 'route': 2,
            'method': 1, 'compound': 2, 'derivative': 2, 'analog': 2,
            'yield': 3, 'reaction': 2, 'mechanism': 1, 'catalyst': 2,
            'reagent': 2, 'chemical': 1, 'organic': 2, 'molecule': 1,
            'drug': 2, 'pharmaceutical': 2, 'medicinal': 2, 'total synthesis': 5
        }
        
        # Irrelevant keywords (negative scoring)
        irrelevant_keywords = {
            'astrophysics': -10, 'galaxy': -8, 'cosmology': -8,
            'economics': -8, 'finance': -8, 'sociology': -8,
            'psychology': -5, 'literature': -5, 'history': -5,
            'computer science': -3  # Can overlap with cheminformatics
        }
        
        # 1. Direct search term match (highest priority)
        if search_lower in title:
            score += 20
        elif search_lower in abstract:
            score += 10
        
        # 2. Chemistry keyword scoring
        for keyword, weight in chemistry_keywords.items():
            if keyword in text_content:
                score += weight
        
        # 3. Irrelevant keyword penalty
        for keyword, penalty in irrelevant_keywords.items():
            if keyword in text_content:
                score += penalty
        
        # 4. Title quality bonus
        synthesis_in_title = any(word in title for word in ['synthesis', 'synthetic', 'preparation'])
        if synthesis_in_title:
            score += 8
        
        # 5. Abstract quality
        if not abstract or len(abstract.split()) < 20:
            score -= 5  # Penalty for very short or missing abstracts
        
        # 6. Chemistry context bonus
        chemistry_context = ['synthesis of', 'preparation of', 'route to', 'formation of']
        if any(context in text_content for context in chemistry_context):
            score += 6
        
        return max(0, score)  # Ensure non-negative score


def search_semantic_scholar(query: str, max_results: int = 10, api_key: str = None) -> List[Dict[str, Any]]:
    """
    Convenience function to search Semantic Scholar
    """
    processor = SemanticScholarProcessor(api_key=api_key)
    return processor.search_papers(query, max_results=max_results)


# Test function
if __name__ == "__main__":
    # Test the fixed processor
    processor = SemanticScholarProcessor()
    
    # Test with a known chemistry compound
    test_queries = [
        "aspirin synthesis",
        "caffeine preparation", 
        "morphine",
        "50-78-2"  # CAS number for aspirin
    ]
    
    for query in test_queries:
        print(f"\n=== Testing query: '{query}' ===")
        results = processor.search_papers(query, max_results=3)
        
        if results:
            print(f"Found {len(results)} papers:")
            for i, paper in enumerate(results, 1):
                print(f"\n{i}. {paper.get('title')}")
                print(f"   Authors: {', '.join(paper.get('authors', [])[:3])}")
                print(f"   Year: {paper.get('published')}")
                print(f"   Citations: {paper.get('citation_count', 0)}")
                print(f"   Relevance Score: {paper.get('relevance_score')}")
                print(f"   URL: {paper.get('pdf_url', 'N/A')}")
                if paper.get('summary'):
                    print(f"   Abstract: {paper.get('summary')[:150]}...")
        else:
            print("No results found!")
    
    print("\n=== Test completed ===")