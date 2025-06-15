from utils import arxiv_processor, web_searcher
from utils.pubchem_processor import search_pubchem_literature
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

def extract_common_name(query: str) -> str:
    """
    Extract common drug name from complex queries.
    This is a simple heuristic - you might want to make this more sophisticated.
    """
    if '[' in query and ']' in query:
        return query
    return query.strip()

def is_cas_number(query: str) -> bool:
    """Check if the query looks like a CAS registry number"""
    cas_pattern = r'^\d{2,7}-\d{2}-\d$'
    return bool(re.match(cas_pattern, query.strip()))

def create_search_variants(original_query: str, compound_name: str = None, resolved_smiles: str = None) -> dict:
    """
    Create different search variants for different databases.
    
    Args:
        original_query: The user's original query
        compound_name: Common name if available (e.g., "Ibuprofen")
        resolved_smiles: SMILES string if available
    
    Returns:
        Dictionary with search terms for each database
    """
    variants = {
        'arxiv_query': original_query,
        'pubchem_query': original_query,
        'web_query': original_query
    }
    
    # Strategy for CAS numbers
    if is_cas_number(original_query):
        # For CAS numbers, try the compound name first for literature searches
        if compound_name and compound_name.lower() != original_query.lower():
            variants['arxiv_query'] = compound_name
            variants['web_query'] = f"{compound_name} OR {original_query}"  # Try both
        # Keep original CAS for PubChem as it handles CAS numbers well
        variants['pubchem_query'] = original_query
    
    # Strategy for other identifiers
    elif compound_name and compound_name.lower() != original_query.lower():
        variants['pubchem_query'] = compound_name
        variants['arxiv_query'] = compound_name
        variants['web_query'] = compound_name
    
    # Additional fallback strategies
    if resolved_smiles and len(resolved_smiles) < 100:  # Only for reasonable length SMILES
        # Don't use SMILES for arXiv (not useful) but keep for PubChem as backup
        pass
    
    return variants

def get_alternative_names(original_query: str, compound_name: str = None) -> list:
    """
    Generate alternative names/terms to try if initial search fails
    """
    alternatives = []
    
    # If we have a compound name, try variations
    if compound_name:
        alternatives.append(compound_name)
        
        # Try without prefixes like "2,5-dichloro" -> "benzoxazole"
        if ',' in compound_name and '-' in compound_name:
            # Extract base compound name
            parts = compound_name.split('-')
            if len(parts) > 1:
                base_name = parts[-1]
                if len(base_name) > 4:  # Reasonable base name
                    alternatives.append(base_name)
        
        # Try partial matches for complex names
        if len(compound_name) > 15:
            words = compound_name.replace('-', ' ').split()
            for word in words:
                if len(word) > 5:  # Meaningful chemical terms
                    alternatives.append(word)
    
    # Add original query if it's different
    if original_query not in alternatives:
        alternatives.append(original_query)
    
    return alternatives

def perform_literature_search(query: str, compound_name: str = None, resolved_smiles: str = None, max_results: int = 5) -> dict:
    """
    Orchestrates a literature search across arXiv, PubChem, and Web sources.
    
    Args:
        query: The search term (could be IUPAC name, SMILES, CAS number, etc.)
        compound_name: Common/trade name if available (e.g., "Ibuprofen")
        resolved_smiles: SMILES string if available
        max_results: The number of results to fetch from each source.

    Returns:
        A dictionary containing lists of results from each source.
    """
    results = {
        "arxiv_papers": [],
        "pubchem_papers": [],
        "web_results": [],
        "errors": []
    }

    # Create search variants
    search_variants = create_search_variants(query, compound_name, resolved_smiles)
    
    print(f"Starting literature search:")
    print(f"  Original query: '{query}'")
    print(f"  Common name: '{compound_name}'")
    print(f"  Resolved SMILES: '{resolved_smiles[:50]}...' " if resolved_smiles and len(resolved_smiles) > 50 else f"  Resolved SMILES: '{resolved_smiles}'")
    print(f"  arXiv query: '{search_variants['arxiv_query']}'")
    print(f"  PubChem query: '{search_variants['pubchem_query']}'")
    print(f"  Web query: '{search_variants['web_query']}'")

    def safe_arxiv_search():
        try:
            # Try primary search first
            results_primary = arxiv_processor.search_arxiv(search_variants['arxiv_query'], max_results=max_results)
            if results_primary:
                return results_primary
            
            # If no results and we have alternatives, try them
            alternatives = get_alternative_names(query, compound_name)
            for alt_name in alternatives[:2]:  # Try max 2 alternatives
                if alt_name != search_variants['arxiv_query']:
                    print(f"Trying arXiv alternative search: '{alt_name}'")
                    alt_results = arxiv_processor.search_arxiv(alt_name, max_results=max_results)
                    if alt_results:
                        return alt_results
            
            return []
        except Exception as e:
            print(f"arXiv search failed: {e}")
            results['errors'].append(f"arXiv search failed: {str(e)}")
            return []

    def safe_pubchem_search():
        try:
            # Try primary search first
            results_primary = search_pubchem_literature(search_variants['pubchem_query'], max_results=max_results)
            if results_primary:
                return results_primary
            
            # If no results, try alternatives
            alternatives = get_alternative_names(query, compound_name)
            for alt_name in alternatives[:3]:  # Try max 3 alternatives for PubChem
                if alt_name != search_variants['pubchem_query']:
                    print(f"Trying PubChem alternative search: '{alt_name}'")
                    alt_results = search_pubchem_literature(alt_name, max_results=max_results)
                    if alt_results:
                        return alt_results
            
            return []
        except Exception as e:
            print(f"PubChem search failed: {e}")
            results['errors'].append(f"PubChem search failed: {str(e)}")
            return []

    def safe_web_search():
        try:
            # For web search, include multiple terms if available
            web_query = search_variants['web_query']
            
            # Enhance web query with additional context
            if is_cas_number(query):
                web_query += " synthesis OR preparation OR chemistry"
            elif compound_name:
                web_query += " chemical OR synthesis OR pharmaceutical"
            
            return web_searcher.search_duckduckgo(web_query, max_results=max_results)
        except Exception as e:
            print(f"Web search failed: {e}")
            results['errors'].append(f"Web search failed: {str(e)}")
            return []

    # Execute searches with better error handling
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_source = {
            executor.submit(safe_arxiv_search): "arxiv",
            executor.submit(safe_pubchem_search): "pubchem", 
            executor.submit(safe_web_search): "web"
        }

        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                data = future.result()
                print(f"{source} search returned {len(data)} results")
                
                if source == "arxiv":
                    results["arxiv_papers"] = data
                elif source == "pubchem":
                    results["pubchem_papers"] = data
                elif source == "web":
                    results["web_results"] = data
                    
            except Exception as exc:
                error_msg = f'{source} search generated an exception: {exc}'
                print(error_msg)
                results['errors'].append(error_msg)

    # Combine and deduplicate papers
    all_papers = results.get("arxiv_papers", []) + results.get("pubchem_papers", [])
    
    if not all_papers:
        print("No papers found from arXiv or PubChem")
        if results['errors']:
            print("Errors encountered:")
            for error in results['errors']:
                print(f"  - {error}")
    
    seen_titles = set()
    unique_papers = []
    for paper in all_papers:
        title = paper.get('title')
        if title and isinstance(title, str) and title.lower() not in seen_titles:
            unique_papers.append(paper)
            seen_titles.add(title.lower())

    formatted_papers = []
    for paper in unique_papers:
        source_name = paper.get("source_db", "N/A") 
        
        formatted_papers.append({
            "title": paper.get("title"),
            "source": f"{', '.join(paper.get('authors', [])[:3])}... ({str(paper.get('published', 'N/A'))[:4]}) - {source_name}",
            "impact": paper.get("primary_category", "N/A"),
            "citations": paper.get("comment", ""),
            "abstract": paper.get("summary"),
            "keywords": paper.get("categories", [])
        })

    print(f"Literature search complete. Found {len(formatted_papers)} unique papers.")
    print(f"Breakdown: {len(results.get('arxiv_papers', []))} arXiv, {len(results.get('pubchem_papers', []))} PubChem")
    
    return {
        "papers": formatted_papers,
        "search_info": {
            "original_query": query,
            "compound_name": compound_name,
            "resolved_smiles": resolved_smiles,
            "search_variants": search_variants,
            "errors": results['errors']
        }
    }