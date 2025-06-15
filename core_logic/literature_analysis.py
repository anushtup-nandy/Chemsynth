from utils import arxiv_processor, web_searcher
from utils.pubchem_processor import search_pubchem_literature
from core_logic.synthesis_planner import resolve_molecule_identifier
from concurrent.futures import ThreadPoolExecutor, as_completed

def perform_literature_search(query: str, max_results: int = 5) -> dict:
    """
    Orchestrates a literature search across multiple sources. It first attempts to
    identify if the query is a chemical. If so, it queries all sources.
    If not, it skips the PubChem chemical literature search.

    Args:
        query: The search term from the user (can be a chemical or a topic).
        max_results: The number of results to fetch from each source.

    Returns:
        A dictionary containing lists of results from each source.
    """
    results = {
        "arxiv_papers": [],
        "pubchem_papers": [],
        "web_results": [],
        "error": None
    }

    resolved_smiles = resolve_molecule_identifier(query)
    
    search_tasks = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        search_tasks[executor.submit(arxiv_processor.search_arxiv, query, max_results=max_results)] = "arxiv"
        search_tasks[executor.submit(web_searcher.search_duckduckgo, query, max_results=max_results)] = "web"

        if resolved_smiles:
            print(f"Query '{query}' identified as a chemical. Including PubChem literature search.")
            search_tasks[executor.submit(search_pubchem_literature, query, max_results=max_results)] = "pubchem"
        else:
            print(f"Query '{query}' appears to be a general topic. Skipping PubChem-specific search.")

        for future in as_completed(search_tasks):
            source = search_tasks[future]
            try:
                data = future.result()
                if source == "arxiv":
                    results["arxiv_papers"] = data
                elif source == "pubchem":
                    results["pubchem_papers"] = data
                elif source == "web":
                    results["web_results"] = data
            except Exception as exc:
                print(f'{source} search generated an exception: {exc}')
                results['error'] = f"Failed to fetch results from {source}."

    all_papers = results.get("arxiv_papers", []) + results.get("pubchem_papers", [])
    
    seen_titles = set()
    unique_papers = []
    for paper in all_papers:
        title = paper.get('title')
        if title and title.lower() not in seen_titles:
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

    return {"papers": formatted_papers}