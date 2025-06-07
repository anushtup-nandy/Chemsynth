# core_logic/literature_analysis.py

from utils import arxiv_processor, web_searcher
from concurrent.futures import ThreadPoolExecutor, as_completed

def perform_literature_search(query: str, max_results: int = 5) -> dict:
    """
    Orchestrates a literature search across multiple sources (arXiv, web search)
    in parallel for better performance.

    Args:
        query: The search term from the user.
        max_results: The number of results to fetch from each source.

    Returns:
        A dictionary containing lists of results from each source.
    """
    results = {
        "arxiv_papers": [],
        "web_results": [],
        "error": None
    }
    
    # Using a ThreadPoolExecutor to run searches in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit tasks to the executor
        future_to_source = {
            executor.submit(arxiv_processor.search_arxiv, query, max_results=max_results): "arxiv",
            executor.submit(web_searcher.search_duckduckgo, query, max_results=max_results): "web"
        }

        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                data = future.result()
                if source == "arxiv":
                    results["arxiv_papers"] = data
                elif source == "web":
                    results["web_results"] = data
            except Exception as exc:
                print(f'{source} search generated an exception: {exc}')
                results['error'] = f"Failed to fetch results from {source}."

    # For this MVP, we will directly use the results. In a more advanced version,
    # you would fetch the content of web results (using web_scraper) and then
    # summarize everything (using llm_interface).
    
    # We will simulate the structure seen in the HTML.
    # The actual paper data comes from our search. Let's format it.
    formatted_papers = []
    for paper in results.get("arxiv_papers", []):
        formatted_papers.append({
            "title": paper.get("title"),
            "source": f"{', '.join(paper.get('authors', []))} ({paper.get('published', 'N/A')[:4]}) - ArXiv",
            "impact": paper.get("primary_category", "N/A"), # Placeholder for impact factor
            "citations": paper.get("comment", ""), # Placeholder for citations
            "abstract": paper.get("summary"),
            "keywords": paper.get("categories", [])
        })

    return {"papers": formatted_papers} # Return a structure the frontend can easily parse