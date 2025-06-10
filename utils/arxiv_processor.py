# utils/arxiv_processor.py
import arxiv
from typing import List, Dict, Optional, Any
from .llm_interface import generate_text
from .prompt_loader import format_prompt, get_prompt_template

# <<< NEW: Define relevant categories to filter the search >>>
# This list includes chemistry, materials science, condensed matter physics,
# and relevant sub-categories of quantitative biology and physics.
RELEVANT_ARXIV_CATEGORIES = [
    'chem', 'cond-mat.mtrl-sci', 'cond-mat.soft', 'physics.chem-ph',
    'physics.bio-ph', 'q-bio.BM', 'q-bio.QM'
]

def search_arxiv(
    query: str,
    max_results: int = 5,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
) -> List[Dict[str, Any]]:
    """
    Searches arXiv for papers matching the query, filtered by relevant scientific categories.

    Args:
        query: The search query (e.g., "quantum computing", "author:John Doe").
        max_results: Maximum number of results to return.
        sort_by: arxiv.SortCriterion for ordering results (Relevance, LastUpdatedDate, SubmittedDate).

    Returns:
        A list of dictionaries, where each dictionary contains metadata for an arXiv paper.
        Returns an empty list on error or if no results.
    """
    search_results_list: List[Dict[str, Any]] = []
    try:
        # <<< MODIFIED: Build a category-specific query string >>>
        category_query = " OR ".join([f"cat:{cat}" for cat in RELEVANT_ARXIV_CATEGORIES])
        # Final query combines user's term with our category filter
        # e.g., "(aspirin synthesis) AND (cat:chem OR cat:cond-mat.mtrl-sci)"
        filtered_query = f"({query}) AND ({category_query})"
        
        print(f"Executing filtered arXiv search with query: {filtered_query}")

        search = arxiv.Search(
            query=filtered_query,
            max_results=max_results,
            sort_by=sort_by
        )
        
        client = arxiv.Client()
        results_iterable = client.results(search)

        for res in results_iterable:
            authors_list = [str(author) for author in res.authors]
            categories_list = res.categories
            
            search_results_list.append({
                "entry_id": res.entry_id,
                "title": res.title,
                "authors": authors_list,
                "summary": res.summary,
                "published": res.published.isoformat() if res.published else None,
                "updated": res.updated.isoformat() if res.updated else None,
                "categories": categories_list,
                "doi": res.doi,
                "pdf_url": res.pdf_url,
                "comment": res.comment,
                "journal_ref": res.journal_ref,
                "primary_category": res.primary_category,
                # Add a source field for easy identification in the UI
                "source_db": "arXiv"
            })
        return search_results_list
    except Exception as e:
        print(f"Error during arXiv search for query '{query}': {e}")
        return []

# The summarize_arxiv_paper_abstract function does not need changes.
def summarize_arxiv_paper_abstract(
    paper_metadata: Dict[str, Any],
    llm_model_name: Optional[str] = None, 
    prompt_key: str = "summarize_arxiv_abstract",
    prompt_file: str = "task_specific_prompts.yaml"
) -> Optional[str]:
    # ... (no changes here)
    abstract = paper_metadata.get("summary")
    if not abstract:
        print(f"Error: Paper metadata for '{paper_metadata.get('title', 'Unknown title')}' missing 'summary' (abstract).")
        return None
    system_persona_data = get_prompt_template("scientific_analyzer_persona", "system_prompts.yaml")
    system_instruction = system_persona_data.get("scientific_analyzer_persona") if system_persona_data else None
    formatted_prompt = format_prompt(
        prompt_key,
        prompt_file,
        abstract_text=abstract
    )
    if not formatted_prompt:
        print(f"Error: Could not format prompt '{prompt_key}' for paper '{paper_metadata.get('title', 'N/A')}'")
        return None
    model_args = {}
    if llm_model_name:
        model_args['model_name'] = llm_model_name
    summary = generate_text(
        prompt=formatted_prompt,
        system_instruction=system_instruction,
        **model_args
    )
    return summary