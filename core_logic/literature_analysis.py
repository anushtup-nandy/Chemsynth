from utils import arxiv_processor, web_searcher
from utils.pubchem_processor import search_pubchem_literature
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.europe_pmc_processor import search_europe_pmc
import re

def is_cas_number(query: str) -> bool:
    """Check if the query looks like a CAS registry number."""
    cas_pattern = r'^\d{2,7}-\d{2}-\d$'
    return bool(re.match(cas_pattern, query.strip()))

def get_search_cascade(original_query: str, compound_name: str = None) -> list[str]:
    """
    Generates a list of search queries, from most specific to most broad.
    
    Args:
        original_query: The user's initial input.
        compound_name: The resolved common name of the compound.

    Returns:
        A list of unique search terms to try in order.
    """
    cascade = []

    # Level 1 & 2: Original query and resolved name
    if original_query:
        cascade.append(original_query.strip())
    if compound_name and compound_name.strip().lower() != original_query.strip().lower():
        cascade.append(compound_name.strip())

    # Level 3: Parent Compound Extraction (Heuristic)
    # This is a simple heuristic. A more advanced version might use SMARTS patterns.
    base_name_to_search = compound_name or original_query
    # Regex to find common chemical parent names (often ending in -ole, -ine, -ene, etc.)
    # It strips prefixes like "2,5-dichloro-", "N-methyl-", etc.
    match = re.search(r'([a-zA-Z]{4,})', base_name_to_search.replace('-', ' '))
    if match:
        parent_compound = match.group(1)
        # Avoid generic terms like 'acid', 'methyl'
        if parent_compound and len(parent_compound) > 4 and parent_compound.lower() not in ['acid', 'methyl', 'ethyl', 'propyl', 'phenyl']:
             # Example: "2,5-dichlorobenzoxazole" -> "dichlorobenzoxazole" -> "benzoxazole"
            if 'chloro' in parent_compound: parent_compound = parent_compound.replace('chloro', '')
            if 'bromo' in parent_compound: parent_compound = parent_compound.replace('bromo', '')
            if 'di' in parent_compound: parent_compound = parent_compound.replace('di', '')
            if 'tri' in parent_compound: parent_compound = parent_compound.replace('tri', '')
            cascade.append(parent_compound)

    # Level 4: General Class Synthesis
    if len(cascade) > 1: # If we have a parent/common name
        base_term = cascade[1] # Use the common name
        cascade.append(f'"{base_term}" synthesis')
        cascade.append(f'"{base_term}" preparation')
        
    # Level 5: Broad Web Search Query
    web_query_terms = [term for term in [original_query, compound_name] if term]
    web_query = " OR ".join(f'"{term}"' for term in web_query_terms)
    web_query += " synthesis OR reaction OR mechanism"
    
    # Remove duplicates while preserving order
    seen = set()
    unique_cascade = [x for x in cascade if not (x in seen or seen.add(x))]
    
    print(f"Generated search cascade: {unique_cascade}")
    
    return unique_cascade, web_query

def perform_literature_search(query: str, compound_name: str = None, resolved_smiles: str = None, max_results: int = 5) -> dict:
    """
    Orchestrates a literature search using a hierarchical cascade strategy.
    """
    results = {
        "arxiv_papers": [],
        "pubchem_papers": [],
        "web_results": [],
        "errors": [],
        "search_log": [] # To track what was searched
    }

    # Generate the cascade of search terms
    search_cascade, web_query = get_search_cascade(query, compound_name)
    
    print(f"Starting literature search for: '{query}'")
    
    # --- Function to search a single source with the cascade ---
    def search_source_with_cascade(search_function, source_name):
        log_entry = f"Searching {source_name}..."
        print(log_entry)
        results['search_log'].append(log_entry)
        
        for i, term in enumerate(search_cascade):
            log_entry = f"  - Level {i+1} attempt: '{term}'"
            print(log_entry)
            results['search_log'].append(log_entry)
            try:
                # For arXiv, use extended categories on later attempts
                use_extended = i > 0
                if source_name == "arXiv":
                    found_results = search_function(term, max_results=max_results, use_extended_categories=use_extended)
                else: # PubChem
                    found_results = search_function(term, max_results=max_results)

                if found_results:
                    log_entry = f"  => Success! Found {len(found_results)} results for '{term}'."
                    print(log_entry)
                    results['search_log'].append(log_entry)
                    return found_results
            except Exception as e:
                error_msg = f"  - Error searching {source_name} for '{term}': {e}"
                print(error_msg)
                results['errors'].append(error_msg)
        
        log_entry = f"  => {source_name} search exhausted, no results found."
        print(log_entry)
        results['search_log'].append(log_entry)
        return []

    # --- Web search uses its own optimized query ---
    def safe_web_search():
        try:
            log_entry = f"Searching Web with query: '{web_query}'"
            print(log_entry)
            results['search_log'].append(log_entry)
            return web_searcher.search_duckduckgo(web_query, max_results=max_results)
        except Exception as e:
            error_msg = f"Web search failed: {e}"
            print(error_msg)
            results['errors'].append(error_msg)
            results['search_log'].append(error_msg)
            return []

    # Execute searches in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_source = {
           executor.submit(search_source_with_cascade, arxiv_processor.search_arxiv, "arXiv"): "arxiv",
            executor.submit(search_source_with_cascade, search_pubchem_literature, "PubChem"): "pubchem",
            executor.submit(search_source_with_cascade, search_europe_pmc, "EuropePMC"): "europe_pmc", # NEW
            executor.submit(safe_web_search): "web"
        }

        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                data = future.result()
                if source == "arxiv": results["arxiv_papers"] = data
                elif source == "pubchem": results["pubchem_papers"] = data
                elif source == "europe_pmc": results["europe_pmc_papers"] = data # NEW
                elif source == "web": results["web_results"] = data
            except Exception as exc:
                error_msg = f'{source} search generated an exception: {exc}'
                print(error_msg)
                results['errors'].append(error_msg)

    # --- Combine and format results ---
    all_papers = results.get("arxiv_papers", []) + results.get("pubchem_papers", []) + results.get("europe_pmc_papers", [])
    seen_titles = set()
    unique_papers = []
    for paper in all_papers:
        title = paper.get('title')
        if title and isinstance(title, str) and title.lower() not in seen_titles:
            unique_papers.append(paper)
            seen_titles.add(title.lower())
    
    formatted_papers = [
        {
            "title": p.get("title"),
            "source": f"{', '.join(p.get('authors', [])[:3])}... ({str(p.get('published', 'N/A'))[:4]}) - {p.get('source_db', 'N/A')}",
            "abstract": p.get("summary"),
            "url": p.get("pdf_url")
        } for p in unique_papers
    ]

    print(f"Literature search complete. Found {len(formatted_papers)} unique papers.")
    
    return {
        "papers": formatted_papers,
        "web_results": results.get("web_results", []), # Also return web results
        "search_info": {
            "original_query": query,
            "compound_name": compound_name,
            "search_cascade_attempted": search_cascade,
            "web_query_used": web_query,
            "search_log": results['search_log'],
            "errors": results['errors']
        }
    }