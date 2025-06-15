# utils/arxiv_processor.py
import arxiv
from typing import List, Dict, Optional, Any
from .llm_interface import generate_text
from .prompt_loader import format_prompt, get_prompt_template

# utils/arxiv_processor.py - Updated categories section

# More focused categories for chemistry and pharmaceutical research
CHEMISTRY_FOCUSED_CATEGORIES = [
    # ==================== PRIMARY CHEMISTRY CATEGORIES ====================
    'chem-ph',  # Chemical Physics - PRIMARY CATEGORY for chemistry
    
    # ==================== PHYSICS - Chemical & Molecular Systems (Most Relevant) ====================
    'physics.chem-ph',    # Chemical Physics
    'physics.bio-ph',     # Biological Physics (drug interactions, molecular biology)
    'physics.comp-ph',    # Computational Physics (DFT, molecular dynamics)
    'physics.data-an',    # Data Analysis (spectroscopy, chemical analysis)
    'physics.atom-ph',    # Atomic Physics (atomic-level processes)
    
    # ==================== CONDENSED MATTER - Materials & Drug Delivery ====================
    'cond-mat.mtrl-sci',  # Materials Science (drug delivery materials, pharmaceutical materials)
    'cond-mat.soft',      # Soft Condensed Matter (polymers, drug delivery systems)
    'cond-mat.mes-hall',  # Mesoscale Physics (nanoparticle drug delivery)
    'cond-mat.stat-mech', # Statistical Mechanics (reaction kinetics)
    
    # ==================== QUANTITATIVE BIOLOGY - Drug Discovery & Biochemistry ====================
    'q-bio.BM',    # Biomolecules (drug targets, protein-drug interactions)
    'q-bio.MN',    # Molecular Networks (metabolic pathways, drug metabolism)
    'q-bio.QM',    # Quantitative Methods (pharmacokinetics, QSAR)
    'q-bio.SC',    # Subcellular Processes (drug action mechanisms)
    'q-bio.TO',    # Tissues and Organs (drug delivery, pharmacology)
    'q-bio.CB',    # Cell Behavior (drug effects on cells)
    
    # ==================== STATISTICS & ML - QSAR and Drug Discovery ====================
    'stat.ML',     # Machine Learning (QSAR, drug discovery ML)
    'stat.AP',     # Applications (clinical trial analysis, experimental design)
    'stat.ME',     # Methodology (statistical methods for pharmaceutical research)
    
    # ==================== MATHEMATICS - Modeling (Limited to Most Relevant) ====================
    'math.OC',     # Optimization (reaction optimization, drug design)
    'math.NA',     # Numerical Analysis (computational chemistry methods)
    'math.DS',     # Dynamical Systems (reaction networks, pharmacokinetics)
]

# Extended categories for broader searches (use when primary search yields few results)
EXTENDED_CHEMISTRY_CATEGORIES = CHEMISTRY_FOCUSED_CATEGORIES + [
    # Additional physics
    'physics.optics',     # Optics (photochemistry, laser applications)
    'physics.app-ph',     # Applied Physics
    'physics.flu-dyn',    # Fluid Dynamics (chemical processes)
    
    # Additional condensed matter
    'cond-mat.dis-nn',    # Disordered Systems
    'cond-mat.other',     # Other Condensed Matter
    
    # Additional quantitative biology
    'q-bio.GN',    # Genomics (pharmacogenomics)
    'q-bio.PE',    # Populations and Evolution
    'q-bio.NC',    # Neurons and Cognition (neurochemistry)
    'q-bio.OT',    # Other Quantitative Biology
    
    # Additional statistics
    'stat.CO',     # Computation
    'stat.TH',     # Theory
    'stat.OT',     # Other Statistics
    
    # Selected mathematics
    'math.PR',     # Probability (stochastic models)
    'math.AP',     # Analysis of PDEs
    'math.SP',     # Spectral Theory
    
    # Quantum physics (for quantum chemistry)
    'quant-ph',    # Quantum Physics
]

def search_arxiv(
    query: str,
    max_results: int = 5,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    use_extended_categories: bool = False
) -> List[Dict[str, Any]]:
    """
    Searches arXiv for papers matching the query, with improved category filtering.

    Args:
        query: The search query
        max_results: Maximum number of results to return
        sort_by: arxiv.SortCriterion for ordering results
        use_extended_categories: Whether to use extended category list for broader search

    Returns:
        A list of dictionaries containing paper metadata
    """
    search_results_list: List[Dict[str, Any]] = []
    
    # Choose category set based on search strategy
    relevant_categories = EXTENDED_CHEMISTRY_CATEGORIES if use_extended_categories else CHEMISTRY_FOCUSED_CATEGORIES
    
    try:   
        print(f"Executing arXiv search with query: '{query}' using {'extended' if use_extended_categories else 'focused'} categories")

        search = arxiv.Search(
            query=query,
            max_results=max_results * 4,  # Fetch more to allow for filtering
            sort_by=sort_by
        )
        
        client = arxiv.Client()
        all_results = list(client.results(search))

        if not all_results:
            print(f"No results found for query: '{query}'")
            return []

        # Filter and prioritize results
        chemistry_papers = []
        other_relevant_papers = []
        other_papers = []

        relevant_categories_set = set(relevant_categories)
        primary_chemistry_categories = {'chem-ph', 'physics.chem-ph', 'q-bio.BM', 'q-bio.MN', 'q-bio.QM'}

        for res in all_results:
            paper_categories = set(res.categories)
            
            # Check relevance level
            if not paper_categories.isdisjoint(primary_chemistry_categories):
                chemistry_papers.append(res)
            elif not paper_categories.isdisjoint(relevant_categories_set):
                other_relevant_papers.append(res)
            else:
                # Check if the title/abstract suggests chemistry relevance
                title_abstract = (res.title + " " + res.summary).lower()
                chemistry_keywords = [
                    'synthesis', 'chemical', 'molecule', 'drug', 'pharmaceutical', 
                    'compound', 'reaction', 'catalyst', 'organic', 'medicinal',
                    'pharmacology', 'therapeutic', 'inhibitor', 'binding',
                    'ibuprofen', 'nsaid', 'anti-inflammatory'
                ]
                
                if any(keyword in title_abstract for keyword in chemistry_keywords):
                    other_papers.append(res)

        # Combine results with priority: chemistry > other relevant > other
        final_results = (chemistry_papers + other_relevant_papers + other_papers)[:max_results]
        
        print(f"Found {len(chemistry_papers)} primary chemistry papers, "
              f"{len(other_relevant_papers)} other relevant papers, "
              f"{len(other_papers)} keyword-matched papers")

        for res in final_results:
            authors_list = [str(author) for author in res.authors]
            
            search_results_list.append({
                "entry_id": res.entry_id,
                "title": res.title,
                "authors": authors_list,
                "summary": res.summary,
                "published": res.published.isoformat() if res.published else None,
                "updated": res.updated.isoformat() if res.updated else None,
                "categories": res.categories,
                "doi": res.doi,
                "pdf_url": res.pdf_url,
                "comment": res.comment,
                "journal_ref": res.journal_ref,
                "primary_category": res.primary_category,
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