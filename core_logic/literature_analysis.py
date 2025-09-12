from utils import arxiv_processor, web_searcher
from utils.pubchem_processor import search_pubchem_literature
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.europe_pmc_processor import search_europe_pmc
import re

def is_cas_number(query: str) -> bool:
    """Check if the query looks like a CAS registry number."""
    cas_pattern = r'^\d{2,7}-\d{2}-\d$'
    return bool(re.match(cas_pattern, query.strip()))

def is_chemical_compound(query: str) -> bool:
    """check if the query looks like a chemical compound"""
    smiles_pattern = r'^[CNOSPFClBrI\[\]()=\-+#@\d]+$'
    formula_pattern = r'^C\d*H\d*[A-Z][a-z]?\d*'

    return (bool(re.match(smiles_pattern, query)) or bool(re.match(formula_pattern, query)) or
            len(query.split()) <= 3)

def get_search_cascade(original_query: str, compound_name: str = None) -> tuple[list[str], str]:
    """
    Generates a list of contextualized search queries, from most specific to most broad.
    """
    cascade = []
    base_terms = []

    # Level 1 & 2: Original query and resolved name
    if original_query:
        base_terms.append(original_query.strip())
    if compound_name and compound_name.strip().lower() != original_query.strip().lower():
        base_terms.append(compound_name.strip())
    
    # Contextualize base terms
    for term in set(base_terms):
        # The query for scholarly databases should be specific
        # cascade.append(f'"{term}" AND (synthesis OR reaction OR chemical OR molecule)')
        cascade.append(f'"{term}" AND (synthesis OR "synthetic route" OR preparation OR "chemical synthesis")')
        cascade.append(term) # Also try without context as a fallback

    # Level 3: Parent Compound Extraction
    base_name_to_search = compound_name or original_query
    match = re.search(r'([a-zA-Z]{5,})', base_name_to_search.replace('-', ' '))
    if match:
        parent_compound = match.group(1).lower()
        if parent_compound not in ['acid', 'chloro', 'bromo', 'di', 'tri']:
            cascade.append(f'"{parent_compound}" synthesis')

    # Create a clean, unique cascade
    seen = set()
    unique_cascade = [x for x in cascade if not (x in seen or seen.add(x))]
    
    # Create the broad web query separately
    web_query = " OR ".join(f'"{term}"' for term in set(base_terms))
    web_query += " synthesis method OR preparation OR reaction mechanism"
    
    print(f"Generated search cascade: {unique_cascade}")
    return unique_cascade, web_query

def perform_literature_search(query: str, compound_name: str = None, resolved_smiles: str = None, max_results: int = 5) -> dict:
    """
    Orchestrates a literature search using a hierarchical cascade strategy.
    """
    results = {
        "arxiv_papers": [], "pubchem_papers": [], "europe_pmc_papers": [], 
        "web_results": [], "errors": [], "search_log": []
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
                    found_results = search_function(term, max_results=max_results* 2, use_extended_categories=use_extended)
                else: # PubChem
                    found_results = search_function(term, max_results=max_results* 2)

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

    def final_relevance_check(papers, original_query):
        """Final check to ensure papers are actually about the compound"""
        validated_papers = []
        query_terms = set(original_query.lower().split())
        
        for paper in papers:
            title_words = set(paper.get('title', '').lower().split())
            abstract_words = set(paper.get('summary', '')[:200].lower().split())  # First 200 chars
            
            # Check if compound name appears in title or early abstract
            if (query_terms & title_words) or (query_terms & abstract_words):
                validated_papers.append(paper)
            elif paper.get('relevance_score', 0) >= 8:  # Keep very high scoring papers
                validated_papers.append(paper)
        
        return validated_papers

    # --- Combine and format results ---
    all_papers_raw = (
        results.get("arxiv_papers", []) + 
        results.get("pubchem_papers", []) + 
        results.get("europe_pmc_papers", [])
    )

    # 1. Deduplicate first to avoid scoring the same paper multiple times
    seen_ids = set()
    unique_papers_raw = []
    for paper in all_papers_raw:
        # Use DOI or entry_id as a unique identifier
        paper_id = paper.get('doi') or paper.get('entry_id')
        if paper_id and paper_id not in seen_ids:
            unique_papers_raw.append(paper)
            seen_ids.add(paper_id)

    # 2. Score and filter the unique papers
    relevant_papers = score_and_filter_results(unique_papers_raw, search_cascade)
    relevant_papers = final_relevance_check(relevant_papers, query)


    print(f"Found {len(all_papers_raw)} raw results, "
          f"filtered down to {len(relevant_papers)} relevant papers after scoring.")

    # 3. Format the final, relevant papers
    formatted_papers = [
        {
            "title": p.get("title"),
            "source": f"{', '.join(p.get('authors', [])[:3])}... ({str(p.get('published', 'N/A'))[:4]}) - {p.get('source_db', 'N/A')}",
            "abstract": p.get("summary"),
            "url": p.get("pdf_url"),
            "relevance_score": p.get("relevance_score") # Include for transparency
        } for p in relevant_papers[:max_results] # Return top N results
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

def score_and_filter_results(
    all_papers: list, 
    search_terms: list, 
    min_relevance_score: int = 5
) -> list:
    """
    Scores papers based on chemical relevance and filters out irrelevant results.

    Args:
        all_papers: A list of combined paper dictionaries from all sources.
        search_terms: The list of terms used in the search cascade (for scoring).
        min_relevance_score: The minimum score required to keep a paper.

    Returns:
        A sorted list of relevant paper dictionaries.
    """
    scored_papers = []
    
    # Define chemistry-related keywords and irrelevant field keywords
    CHEM_KEYWORDS = {
        'synthesis', 'synthetic', 'preparation', 'route', 'method', 
        'compound', 'derivative', 'analog', 'scaffold', 'precursor',
        'yield', 'reaction', 'mechanism', 'catalyst', 'reagent'
    }
    IRRELEVANT_KEYWORDS = {
        'astrophysics', 'galaxy', 'cosmology', 'black hole', 'star', 'nebula',
        'economics', 'finance', 'market', 'sociology', 'gravity', 'quantum field',
        'string theory', 'mathematics' # Use 'mathematics' carefully, can overlap
    }

    def contains_compound_reference(text: str, search_terms: list) -> bool:
        """Check if text actually discusses the specific compound"""
        text_lower = text.lower()
        compound_mentioned = any(term.lower() in text_lower for term in search_terms[:2])
        
        # Look for synthesis-related context around the compound mention
        if compound_mentioned:
            synthesis_context = ['synthesis of', 'preparation of', 'route to', 'making', 'formation of']
            return any(context in text_lower for context in synthesis_context)
        return False

    # Use the first two (most specific) search terms for primary scoring
    primary_search_terms = [term.lower() for term in search_terms[:2]]

    for paper in all_papers:
        relevance_score = 0
        title = paper.get('title', '').lower()
        abstract = paper.get('summary', '').lower()
        text_content = title + " " + abstract

        # --- Scoring Logic ---

        # 1. Direct hit on primary search term (highest value)
        if any(term in title for term in primary_search_terms):
            relevance_score += 10
        elif any(term in abstract for term in primary_search_terms):
            relevance_score += 5

        # 2. Source credibility bonus
        source_db = paper.get('source_db', '')
        if 'PubChem' in source_db or 'Europe PMC' in source_db:
            relevance_score += 3  # These are highly relevant sources

        # 3. Chemistry keyword bonus
        found_chem_keywords = len(CHEM_KEYWORDS.intersection(text_content.split()))
        relevance_score += found_chem_keywords

        # 4. arXiv category bonus/penalty
        if source_db == 'arXiv':
            categories = paper.get('categories', [])
            if any(cat.startswith('chem') or cat.startswith('cond-mat.mtrl-sci') or cat.startswith('q-bio') for cat in categories):
                relevance_score += 5
            # Penalize physics categories that are not chem-related
            if any(cat.startswith('astro-ph') or cat.startswith('gr-qc') or cat.startswith('hep-') for cat in categories):
                relevance_score -= 20
            if 'math-ph' in categories or 'math.' in str(categories):
                relevance_score -= 10

        # 5. Penalty for irrelevant keywords
        found_irrelevant_keywords = len(IRRELEVANT_KEYWORDS.intersection(text_content.split()))
        relevance_score -= found_irrelevant_keywords * 5

        # --- Final Decision ---
        if not contains_compound_reference(text_content, search_terms[:2]):
            continue 
        if relevance_score >= min_relevance_score:
            paper['relevance_score'] = relevance_score # Add score to dict for debugging
            scored_papers.append(paper)

    # Sort by relevance score, descending
    return sorted(scored_papers, key=lambda x: x['relevance_score'], reverse=True)