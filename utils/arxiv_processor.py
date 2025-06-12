# utils/arxiv_processor.py
import arxiv
from typing import List, Dict, Optional, Any
from .llm_interface import generate_text
from .prompt_loader import format_prompt, get_prompt_template

# <<< NEW: Define relevant categories to filter the search >>>
# This list includes chemistry, materials science, condensed matter physics,
# and relevant sub-categories of quantitative biology and physics.
RELEVANT_ARXIV_CATEGORIES = [
    # ==================== CHEMISTRY (All Categories) ====================
    'chem-ph',  # Chemical Physics - PRIMARY CATEGORY for chemistry
    
    # ==================== PHYSICS - Chemical & Molecular Systems ====================
    'physics.chem-ph',    # Chemical Physics
    'physics.bio-ph',     # Biological Physics (biochemical synthesis, molecular interactions)
    'physics.comp-ph',    # Computational Physics (DFT, molecular dynamics, quantum chemistry)
    'physics.data-an',    # Data Analysis (spectroscopy, chemical data analysis)
    'physics.atom-ph',    # Atomic Physics (atomic-level chemical processes)
    'physics.optics',     # Optics (photochemistry, laser-assisted synthesis)
    'physics.plasm-ph',   # Plasma Physics (plasma-enhanced synthesis)
    'physics.app-ph',     # Applied Physics (applied chemical processes)
    'physics.flu-dyn',    # Fluid Dynamics (chemical reactions in flow systems)
    'physics.class-ph',   # Classical Physics (thermodynamics of chemical systems)
    
    # ==================== CONDENSED MATTER - Materials & Molecular Systems ====================
    'cond-mat.mtrl-sci',  # Materials Science (synthesis of new materials)
    'cond-mat.soft',      # Soft Condensed Matter (polymers, colloids, drug delivery)
    'cond-mat.mes-hall',  # Mesoscale and Nanoscale Physics (nanoparticle synthesis)
    'cond-mat.str-el',    # Strongly Correlated Electrons (electronic materials synthesis)
    'cond-mat.supr-con',  # Superconductivity (superconducting material synthesis)
    'cond-mat.dis-nn',    # Disordered Systems (amorphous materials, glasses)
    'cond-mat.stat-mech', # Statistical Mechanics (reaction kinetics, phase transitions)
    'cond-mat.other',     # Other Condensed Matter (miscellaneous materials research)
    
    # ==================== QUANTITATIVE BIOLOGY - Comprehensive Coverage ====================
    'q-bio.BM',    # Biomolecules (protein synthesis, nucleic acid synthesis)
    'q-bio.MN',    # Molecular Networks (metabolic pathways, biosynthesis networks)
    'q-bio.QM',    # Quantitative Methods (pharmacokinetics, systems biology)
    'q-bio.SC',    # Subcellular Processes (enzymatic synthesis, cellular metabolism)
    'q-bio.TO',    # Tissues and Organs (tissue engineering, biomedical synthesis)
    'q-bio.CB',    # Cell Behavior (cell-based synthesis, biotechnology)
    'q-bio.GN',    # Genomics (genetic engineering, synthetic biology)
    'q-bio.PE',    # Populations and Evolution (chemical ecology, evolutionary biochemistry)
    'q-bio.NC',    # Neurons and Cognition (neurotransmitter synthesis, neurochemistry)
    'q-bio.OT',    # Other Quantitative Biology (interdisciplinary bio-chem research)
    
    # ==================== COMPUTER SCIENCE - AI/ML/Computational Chemistry ====================
    'cs.LG',       # Machine Learning (ML for synthesis prediction, retrosynthesis)
    'cs.AI',       # Artificial Intelligence (AI-driven drug discovery, synthesis planning)
    'cs.CE',       # Computational Engineering (molecular simulation, process optimization)
    'cs.CV',       # Computer Vision (molecular structure recognition, spectral analysis)
    'cs.IR',       # Information Retrieval (chemical database mining, literature analysis)
    'cs.DC',       # Distributed Computing (high-throughput virtual screening)
    'cs.DS',       # Data Structures and Algorithms (graph algorithms for molecules)
    'cs.ET',       # Emerging Technologies (quantum computing for chemistry)
    'cs.HC',       # Human-Computer Interaction (chemical informatics interfaces)
    'cs.IT',       # Information Theory (molecular information processing)
    'cs.NE',       # Neural and Evolutionary Computing (genetic algorithms for synthesis)
    'cs.PL',       # Programming Languages (domain-specific languages for chemistry)
    'cs.RO',       # Robotics (automated synthesis, lab automation)
    'cs.SC',       # Symbolic Computation (computer algebra for chemistry)
    'cs.SE',       # Software Engineering (chemical software development)
    'cs.SY',       # Systems and Control (process control, reaction optimization)
    
    # ==================== STATISTICS - Chemical Data & Experimental Design ====================
    'stat.ML',     # Machine Learning (QSAR, molecular descriptor analysis)
    'stat.AP',     # Applications (experimental design, quality control)
    'stat.ME',     # Methodology (statistical methods for chemical analysis)
    'stat.CO',     # Computation (computational statistics for chemistry)
    'stat.TH',     # Theory (statistical theory relevant to chemical analysis)
    'stat.OT',     # Other Statistics (interdisciplinary statistical applications)
    
    # ==================== MATHEMATICS - Modeling & Optimization ====================
    'math.OC',     # Optimization and Control (reaction optimization, process control)
    'math.NA',     # Numerical Analysis (numerical methods for chemistry)
    'math.PR',     # Probability (stochastic models, reaction kinetics)
    'math.DS',     # Dynamical Systems (chemical reaction networks)
    'math.AP',     # Analysis of PDEs (reaction-diffusion equations)
    'math.SP',     # Spectral Theory (spectroscopic analysis, quantum mechanics)
    'math.CA',     # Classical Analysis (mathematical chemistry foundations)
    'math.DG',     # Differential Geometry (molecular geometry, conformational analysis)
    'math.CO',     # Combinatorics (chemical graph theory, molecular enumeration)
    'math.GM',     # General Mathematics (mathematical chemistry)
    'math.GN',     # General Topology (chemical topology)
    'math.GT',     # Geometric Topology (molecular topology)
    'math.MG',     # Metric Geometry (molecular distance geometry)
    'math.NT',     # Number Theory (applications in crystallography)
    'math.AG',     # Algebraic Geometry (algebraic methods in chemistry)
    'math.AT',     # Algebraic Topology (topological data analysis for molecules)
    'math.CT',     # Category Theory (categorical approaches to chemistry)
    'math.FA',     # Functional Analysis (quantum chemistry, spectral methods)
    'math.GR',     # Group Theory (molecular symmetry, crystallography)
    'math.LO',     # Logic (automated reasoning in chemistry)
    'math.MP',     # Mathematical Physics (quantum chemistry, statistical mechanics)
    'math.QA',     # Quantum Algebra (quantum groups in chemistry)
    'math.RA',     # Rings and Algebras (algebraic structures in chemistry)
    'math.RT',     # Representation Theory (symmetry in molecular systems)
    'math.SG',     # Symplectic Geometry (Hamiltonian mechanics in chemistry)
    'math.ST',     # Statistics Theory (theoretical statistics for chemistry)
    
    # ==================== NUCLEAR & HIGH ENERGY PHYSICS (Relevant Subsets) ====================
    'nucl-ex',     # Nuclear Experiment (radiochemistry, nuclear synthesis)
    'nucl-th',     # Nuclear Theory (nuclear reaction mechanisms)
    'hep-ex',      # High Energy Physics - Experiment (particle detection chemistry)
    'hep-ph',      # High Energy Physics - Phenomenology (fundamental interactions)
    
    # ==================== ASTROPHYSICS (Astrochemistry) ====================
    'astro-ph.EP', # Earth and Planetary Astrophysics (planetary chemistry)
    'astro-ph.GA', # Astrophysics of Galaxies (interstellar chemistry)
    'astro-ph.SR', # Solar and Stellar Astrophysics (stellar nucleosynthesis)
    'astro-ph.IM', # Instrumentation and Methods (astronomical spectroscopy)
    
    # ==================== NONLINEAR SCIENCES ====================
    'nlin.AO',     # Adaptation and Self-Organizing Systems (self-assembly)
    'nlin.CD',     # Chaotic Dynamics (chaos in chemical reactions)
    'nlin.PS',     # Pattern Formation (chemical patterns, Turing patterns)
    
    # ==================== ECONOMICS (Relevant to Chemical Industry) ====================
    'econ.GN',     # General Economics (chemical industry economics)
    
    # ==================== ELECTRICAL ENGINEERING & SYSTEMS SCIENCE ====================
    'eess.AS',     # Audio and Speech Processing (chemical acoustic analysis)
    'eess.IV',     # Image and Video Processing (microscopy, spectral imaging)
    'eess.SP',     # Signal Processing (chemical signal analysis, NMR processing)
    'eess.SY',     # Systems and Control (chemical process control)
    
    # ==================== ADDITIONAL INTERDISCIPLINARY CATEGORIES ====================
    # These categories often contain chemical synthesis research
    'quant-ph',    # Quantum Physics (quantum chemistry, quantum materials)
    'gr-qc',       # General Relativity (cosmochemistry, extreme conditions)
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