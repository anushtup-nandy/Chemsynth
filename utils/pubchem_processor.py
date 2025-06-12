# utils/pubchem_processor.py

import pubchempy as pcp
from typing import List, Dict, Any
import time

def search_pubchem_literature(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches PubChem for literature associated with a chemical identifier.
    This version uses a more robust method for identifier resolution and literature retrieval.

    Args:
        query: The search query (chemical name, SMILES, InChI, CAS, etc.).
        max_results: The maximum number of literature results to return.

    Returns:
        A list of dictionaries, where each dictionary contains metadata for a paper.
        Returns an empty list on error or if no results.
    """
    print(f"Searching PubChem literature for: {query}")
    literature_results: List[Dict[str, Any]] = []
    try:
        # REVISED STEP 1: Use robust, auto-detecting identifier resolution.
        # By removing the 'name' namespace, pubchempy will automatically detect if the
        # query is a name, SMILES, InChI, CID, etc. This is far more flexible.
        compounds = pcp.get_compounds(query, listkey_count=1)
        if not compounds:
            print(f"Could not resolve '{query}' to a PubChem compound.")
            return []
        
        target_compound = compounds[0]
        cid = target_compound.cid
        print(f"Resolved '{query}' to compound '{target_compound.iupac_name}' (CID: {cid})")

        # REVISED STEP 2: Fetch PubMed IDs (PMIDs) directly from the compound object.
        # The `compound.pmids` attribute is the canonical, reliable way to get this data.
        # It returns a list of integers, which is much cleaner than parsing a string.
        pmid_list = target_compound.pmids
        if not pmid_list:
            print(f"No PubMed articles found for CID {cid} ({query}).")
            return []
            
        print(f"Found {len(pmid_list)} associated PMIDs. Fetching details for up to {max_results}...")
        
        # STEP 3: Fetch details for each PMID. This part remains largely the same,
        # but we now iterate over a clean list of integer PMIDs.
        for pmid in pmid_list[:max_results]:
            # The 'entrez' system is the backend for PubMed.
            results = pcp.Entrez.esummary(db='pubmed', id=pmid)
            for res in results:
                # Normalize the data to match our application's unified data model.
                authors = [author['name'] for author in res.authors if 'name' in author]
                
                # Construct a URL to the paper, preferring DOI.
                doi = res.elocation_id if res.elocation_id and 'doi' in res.elocation_id else None
                pdf_url = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{res.uid}/"

                literature_results.append({
                    "entry_id": f"PMID:{res.uid}",
                    "title": res.title,
                    "authors": authors,
                    # The 'abstract' is often a direct attribute. Fallback to empty.
                    "summary": res.abstract or "No abstract available.",
                    "published": res.pubdate,
                    "updated": res.epubdate, # epubdate is a better 'updated' field
                    # We can extract MeSH terms for better categorization
                    "categories": [term['term'] for term in res.mesh_headings],
                    "doi": doi,
                    "pdf_url": pdf_url,
                    "comment": f"PMID: {res.uid}",
                    "journal_ref": res.fulljournalname,
                    "primary_category": "Pharmacology", # More specific default
                    # Add a source field for easy identification in the UI
                    "source_db": "PubChem/PubMed" 
                })
            # Rate limit to be a good API citizen (3-5 requests/sec is safe)
            time.sleep(0.3) 

        return literature_results

    except pcp.PubChemHTTPError as e:
        print(f"A PubChem API error occurred during literature search for '{query}': {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during PubChem literature search for '{query}': {e}")
        return []