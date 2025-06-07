# utils/arxiv_processor.py
import arxiv
from typing import List, Dict, Optional, Any
from .llm_interface import generate_text
from .prompt_loader import format_prompt, get_prompt_template

# Note: The 'arxiv' library does not require an API key for basic search and metadata.
# For PDF downloading and processing, additional libraries like PyPDF2 or pdfminer.six would be needed.

def search_arxiv(
    query: str,
    max_results: int = 5,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
) -> List[Dict[str, Any]]:
    """
    Searches arXiv for papers matching the query.

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
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by
        )
        
        client = arxiv.Client() # Default client
        results_iterable = client.results(search)

        for res in results_iterable:
            # The 'res' object is an arxiv.Result object
            authors_list = [str(author) for author in res.authors]
            categories_list = res.categories
            
            search_results_list.append({
                "entry_id": res.entry_id, # URL like http://arxiv.org/abs/2303.08774v1
                "title": res.title,
                "authors": authors_list,
                "summary": res.summary, # This is the abstract
                "published": res.published.isoformat() if res.published else None,
                "updated": res.updated.isoformat() if res.updated else None,
                "categories": categories_list,
                "doi": res.doi,
                "pdf_url": res.pdf_url,
                "comment": res.comment,
                "journal_ref": res.journal_ref,
                "primary_category": res.primary_category
            })
        return search_results_list
    except Exception as e:
        print(f"Error during arXiv search for query '{query}': {e}")
        return []

def summarize_arxiv_paper_abstract(
    paper_metadata: Dict[str, Any],
    llm_model_name: Optional[str] = None, # Allow overriding default LLM
    prompt_key: str = "summarize_arxiv_abstract",
    prompt_file: str = "task_specific_prompts.yaml"
) -> Optional[str]:
    """
    Summarizes the abstract of an arXiv paper using an LLM.

    Args:
        paper_metadata: A dictionary containing paper metadata, must include 'summary' (the abstract).
        llm_model_name: Optional name of the LLM model to use.
        prompt_key: The key for the summarization prompt in the YAML file.
        prompt_file: The YAML file containing the prompt.

    Returns:
        The summarized abstract as a string, or None if summarization fails or abstract is missing.
    """
    abstract = paper_metadata.get("summary")
    if not abstract:
        print(f"Error: Paper metadata for '{paper_metadata.get('title', 'Unknown title')}' missing 'summary' (abstract).")
        return None

    # Get system instruction for the LLM if desired
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

    # Use default LLM model if not specified
    model_args = {}
    if llm_model_name:
        model_args['model_name'] = llm_model_name
    
    summary = generate_text(
        prompt=formatted_prompt,
        system_instruction=system_instruction,
        **model_args
    )
    return summary


if __name__ == '__main__':
    print("Testing ArXiv Processor...")
    
    # Test ArXiv Search
    search_query = "transformer models for natural language processing"
    print(f"\n--- Searching ArXiv for: '{search_query}' (max 2 results) ---")
    papers = search_arxiv(search_query, max_results=2)

    if papers:
        for i, paper in enumerate(papers):
            print(f"\nResult {i+1}:")
            print(f"  Title: {paper['title']}")
            print(f"  Authors: {', '.join(paper['authors'])}")
            print(f"  Published: {paper['published']}")
            print(f"  Abstract (first 200 chars): {paper['summary'][:200]}...")
            print(f"  PDF URL: {paper['pdf_url']}")

        # Test Abstract Summarization (using the first paper found)
        # Ensure your GEMINI_API_KEY is set in .env for this to work
        print("\n--- Summarizing Abstract of the first paper ---")
        first_paper = papers[0]
        
        # Check if LLM is available (rudimentary check based on previous llm_interface setup)
        try:
            from config import APP_CONFIG
            if not APP_CONFIG.GEMINI_API_KEY:
                print("Skipping LLM summarization: GEMINI_API_KEY not found.")
            else:
                summary = summarize_arxiv_paper_abstract(first_paper)
                if summary:
                    print(f"\nLLM Summary for '{first_paper['title']}':\n{summary}")
                else:
                    print("Failed to generate summary for the abstract.")
        except ImportError:
             print("Skipping LLM summarization: config.py not found (likely standalone run).")

    else:
        print(f"No papers found for query: '{search_query}' or an error occurred.")
    
    print("\n--- Note on PDF Processing ---")
    print("This module currently focuses on metadata and abstract summarization.")
    print("To process full PDF content, you would need to:")
    print("1. Download the PDF (e.g., using 'requests' and paper['pdf_url']).")
    print("2. Extract text from PDF (e.g., using 'PyPDF2', 'pdfminer.six', or cloud services).")
    print("3. Implement chunking strategies if the text is too long for the LLM context window.")