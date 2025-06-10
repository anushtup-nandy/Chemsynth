
import pubchempy as pcp
from typing import List, Dict, Any
import time

def search_pubchem_literature(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches PubChem for literature associated with a chemical identifier.

    Args:
        query: The search query (chemical name, SMILES, etc.).
        max_results: The maximum number of literature results to return.

    Returns:
        A list of dictionaries, where each dictionary contains metadata for a paper.
        Returns an empty list on error or if no results.
    """
    print(f"Searching PubChem literature for: {query}")
    literature_results = []
    try:
        # 1. Resolve the query to a PubChem Compound
        compounds = pcp.get_compounds(query, 'name', listkey_count=1)
        if not compounds:
            print(f"Could not resolve '{query}' to a PubChem compound.")
            return []
        
        target_compound = compounds[0]
        cid = target_compound.cid

        # 2. Fetch PubMed IDs (PMIDs) associated with the compound's CID
        # We query the 'substance' domain for links to literature.
        pmids = pcp.get_properties('PubMedID', {'CID': cid}, 'substance', searchtype='CID')

        if not pmids or 'PubMedID' not in pmids[0]:
            print(f"No PubMed articles found for CID {cid} ({query}).")
            return []
            
        # The result is a list of dicts, and the PMIDs are in a string, comma-separated
        pmid_list = pmids[0]['PubMedID'].split(',')
        
        # 3. Fetch details for each PMID using PubChem's 'pcassay' interface for literature
        # We need to be careful not to spam the API.
        for pmid in pmid_list[:max_results]:
            # The 'entrez' system is the backend for PubMed.
            results = pcp.Entrez.esummary(db='pubmed', id=pmid)
            for res in results:
                # Normalize the data to match our application's format
                authors = [author['name'] for author in res.authors]
                literature_results.append({
                    "entry_id": f"PMID:{res.uid}",
                    "title": res.title,
                    "authors": authors,
                    "summary": res.attributes.get('abstract', 'No abstract available.'),
                    "published": res.pubdate,
                    "updated": res.lastauthor_sort,
                    "categories": ["Biochemistry", "Pharmacology"], # Placeholder categories
                    "doi": res.attributes.get('DOI'),
                    "pdf_url": f"https://doi.org/{res.attributes.get('DOI')}" if res.attributes.get('DOI') else None,
                    "comment": None,
                    "journal_ref": res.fulljournalname,
                    "primary_category": "Chemistry",
                    # Add a source field for easy identification in the UI
                    "source_db": "PubChem/PubMed" 
                })
            time.sleep(0.3) # Rate limit to be a good API citizen

        return literature_results

    except Exception as e:
        print(f"An error occurred during PubChem literature search for '{query}': {e}")
        return []