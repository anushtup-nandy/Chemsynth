# core_logic/literature_analysis.py

from utils import arxiv_processor, web_searcher
from utils.pubchem_processor import search_pubchem_literature
from concurrent.futures import ThreadPoolExecutor, as_completed

def perform_literature_search(query: str, max_results: int = 5) -> dict:
    """
    Orchestrates a literature search across multiple sources (arXiv, PubChem, web)
    in parallel for better performance.

    Args:
        query: The search term from the user.
        max_results: The number of results to fetch from each source.

    Returns:
        A dictionary containing lists of results from each source.
    """
    results = {
        "arxiv_papers": [],
        "pubchem_papers": [], # <<< NEW DICT KEY
        "web_results": [],
        "error": None
    }

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks to the executor
        future_to_source = {
            executor.submit(arxiv_processor.search_arxiv, query, max_results=max_results): "arxiv",
            executor.submit(search_pubchem_literature, query, max_results=max_results): "pubchem",
            executor.submit(web_searcher.search_duckduckgo, query, max_results=max_results): "web"
        }

        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                data = future.result()
                if source == "arxiv":
                    results["arxiv_papers"] = data
                elif source == "pubchem": # <<< NEW RESULT HANDLING
                    results["pubchem_papers"] = data
                elif source == "web":
                    results["web_results"] = data
            except Exception as exc:
                print(f'{source} search generated an exception: {exc}')
                results['error'] = f"Failed to fetch results from {source}."

    # For this MVP, we will directly use the results. In a more advanced version,
    # you would fetch the content of web results (using web_scraper) and then
    # summarize everything (using llm_interface).
    
    # <<< MODIFIED: Consolidate papers from all academic sources (arXiv and PubChem) >>>
    all_papers = results.get("arxiv_papers", []) + results.get("pubchem_papers", [])
    
    # Remove duplicates by title (a simple de-duplication strategy)
    seen_titles = set()
    unique_papers = []
    for paper in all_papers:
        if paper['title'].lower() not in seen_titles:
            unique_papers.append(paper)
            seen_titles.add(paper['title'].lower())

    # Format the combined, unique list of papers for the frontend
    formatted_papers = []
    for paper in unique_papers:
        # Use the specific source from the paper data, or a fallback
        source_name = paper.get("source_db", "N/A") 
        
        formatted_papers.append({
            "title": paper.get("title"),
            # Display authors, publication year, and the database it came from
            "source": f"{', '.join(paper.get('authors', [])[:3])}... ({str(paper.get('published', 'N/A'))[:4]}) - {source_name}",
            "impact": paper.get("primary_category", "N/A"),
            "citations": paper.get("comment", ""),
            "abstract": paper.get("summary"),
            "keywords": paper.get("categories", [])
        })

    return {"papers": formatted_papers}