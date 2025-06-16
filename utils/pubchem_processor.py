import pubchempy as pcp
from typing import List, Dict, Any, Optional
import time
import requests
from urllib.parse import quote

def search_pubchem_literature(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches PubChem for literature associated with a chemical identifier.
    Tries multiple search strategies to improve success rate.

    Args:
        query: The search query (chemical name, SMILES, CAS number, etc.)
        max_results: The maximum number of literature results to return

    Returns:
        A list of dictionaries containing paper metadata
    """
    print(f"Searching PubChem literature for query: '{query}'")
    literature_results: List[Dict[str, Any]] = []
    
    # Enhanced search strategies - include CAS number search
    search_strategies = [
        ('name', query),
        ('smiles', query),
        ('inchi', query),
        ('formula', query)
    ]
    
    if _is_cas_number(query):
        search_strategies.insert(0, ('name', query))  
        print(f"Detected CAS number format: {query}")
    
    compounds = []
    successful_strategy = None
    
    for search_type, search_term in search_strategies:
        try:
            print(f"Trying PubChem search with {search_type}: '{search_term}'")
            compounds = pcp.get_compounds(search_term, search_type, listkey_count=10) 
            if compounds:
                successful_strategy = search_type
                print(f"Success with {search_type} search - found {len(compounds)} compounds")
                break
        except Exception as e:
            print(f"Failed {search_type} search: {e}")
            continue
    
    if not compounds:
        print(f"PubChem could not resolve '{query}' using any search strategy")
        return []
    
    # Try each compound until we find one with literature
    for i, compound in enumerate(compounds):
        try:
            cid = compound.cid
            compound_name = _get_best_compound_name(compound)
            print(f"Checking compound {i+1}/{len(compounds)}: '{compound_name}' (CID: {cid})")

            # Check if the compound has PMIDs - handle the AttributeError
            pmid_list = []
            try:
                pmid_list = compound.pmids
                if pmid_list is None:
                    pmid_list = []
            except AttributeError:
                print(f"Compound CID {cid} does not have pmids attribute - trying alternative method")
                pmid_list = _get_pmids_alternative(cid)
            
            if not pmid_list:
                print(f"No PubMed articles found for CID {cid}")
                continue
                
            print(f"Found {len(pmid_list)} associated PMIDs for CID {cid}")
            
            # Fetch literature details
            literature_results = _fetch_pubmed_details(pmid_list[:max_results], cid, compound_name)
            
            if literature_results:
                print(f"Successfully retrieved {len(literature_results)} papers for CID {cid}")
                return literature_results
                
        except Exception as e:
            print(f"Error processing compound CID {compound.cid}: {e}")
            continue
    
    print(f"No literature found for any compounds matching '{query}'")
    return []

def _is_cas_number(query: str) -> bool:
    """Check if the query looks like a CAS registry number (format: XXXXX-XX-X)"""
    import re
    cas_pattern = r'^\d{2,7}-\d{2}-\d$'
    return bool(re.match(cas_pattern, query.strip()))

def _get_best_compound_name(compound) -> str:
    """Get the best available name for a compound"""
    try:
        if hasattr(compound, 'synonyms') and compound.synonyms:
            return compound.synonyms[0]
    except:
        pass
    
    try:
        # Try IUPAC name
        if hasattr(compound, 'iupac_name') and compound.iupac_name:
            return compound.iupac_name
    except:
        pass
    
    try:
        if hasattr(compound, 'molecular_formula') and compound.molecular_formula:
            return f"Compound with formula {compound.molecular_formula}"
    except:
        pass
    
    return f'CID-{compound.cid}'

def _get_pmids_alternative(cid: int) -> List[int]:
    """
    Alternative method to get PMIDs using PubChem API directly
    """
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/PubMedID/JSON"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'InformationList' in data and 'Information' in data['InformationList']:
                info = data['InformationList']['Information'][0]
                if 'PubMedID' in info:
                    pmids = info['PubMedID']
                    print(f"Found {len(pmids)} PMIDs via direct API for CID {cid}")
                    return [int(pmid) for pmid in pmids]
        
        print(f"No PMIDs found via direct API for CID {cid}")
        return []
        
    except Exception as e:
        print(f"Error getting PMIDs via alternative method for CID {cid}: {e}")
        return []

def _fetch_pubmed_details(pmid_list: List[int], cid: int, compound_name: str) -> List[Dict[str, Any]]:
    """
    Fetch detailed information for PubMed IDs.
    
    Args:
        pmid_list: List of PubMed IDs
        cid: PubChem compound ID
        compound_name: Name of the compound
        
    Returns:
        List of paper metadata dictionaries
    """
    literature_results = []
    
    if not pmid_list:
        return []
    
    try:
        # Use Entrez to fetch paper details
        from Bio import Entrez
        Entrez.email = "your-email@example.com"  # Required by NCBI
        
        # Fetch summaries for all PMIDs at once
        pmid_str = ','.join(map(str, pmid_list))
        handle = Entrez.esummary(db='pubmed', id=pmid_str)
        records = Entrez.read(handle)
        handle.close()
        
        for record in records:
            try:
                # Extract author information
                authors = []
                if 'AuthorList' in record:
                    authors = [author for author in record['AuthorList']]
                elif 'Authors' in record:
                    # Fallback for different record formats
                    authors = record['Authors'].split(', ') if record['Authors'] else []
                
                # Extract DOI
                doi = None
                if 'DOI' in record and record['DOI']:
                    doi = record['DOI']
                elif 'ELocationID' in record:
                    eloc = record['ELocationID']
                    if 'doi:' in eloc:
                        doi = eloc.replace('doi:', '')
                
                # Create paper entry
                paper_entry = {
                    "entry_id": f"PMID:{record['Id']}",
                    "title": record.get('Title', 'No title available'),
                    "authors": authors,
                    "summary": record.get('Abstract', 'No abstract available'),
                    "published": record.get('PubDate', 'Unknown'),
                    "updated": record.get('EPubDate', ''),
                    "categories": [],  # PubMed doesn't have arXiv-style categories
                    "doi": doi,
                    "pdf_url": f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{record['Id']}/",
                    "comment": f"PMID: {record['Id']}, Related to CID: {cid}",
                    "journal_ref": record.get('FullJournalName', record.get('Source', 'Unknown journal')),
                    "primary_category": "Pharmacology/Chemistry",
                    "source_db": "PubChem/PubMed",
                    "compound_cid": cid,
                    "compound_name": compound_name
                }
                
                literature_results.append(paper_entry)
                
            except Exception as e:
                print(f"Error processing PMID {record.get('Id', 'unknown')}: {e}")
                continue
                
        return literature_results
        
    except ImportError:
        print("BioPython not available, using alternative method")
        return _fetch_pubmed_details_alternative(pmid_list, cid, compound_name)
    except Exception as e:
        print(f"Error fetching PubMed details: {e}")
        return []

def _fetch_pubmed_details_alternative(pmid_list: List[int], cid: int, compound_name: str) -> List[Dict[str, Any]]:
    """
    Alternative method to fetch PubMed details without BioPython.
    """
    literature_results = []
    
    for pmid in pmid_list[:5]:  # Limit to avoid rate limiting
        try:
            # Use NCBI's E-utilities directly
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if 'result' in data and str(pmid) in data['result']:
                    record = data['result'][str(pmid)]
                    
                    # Extract authors
                    authors = []
                    if 'authors' in record:
                        authors = [author['name'] for author in record['authors'] if 'name' in author]
                    
                    paper_entry = {
                        "entry_id": f"PMID:{pmid}",
                        "title": record.get('title', 'No title available'),
                        "authors": authors,
                        "summary": "Abstract not available in summary",
                        "published": record.get('pubdate', 'Unknown'),
                        "updated": record.get('epubdate', ''),
                        "categories": [],
                        "doi": record.get('elocationid', '').replace('doi: ', '') if 'doi:' in record.get('elocationid', '') else None,
                        "pdf_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "comment": f"PMID: {pmid}, Related to CID: {cid}",
                        "journal_ref": record.get('fulljournalname', 'Unknown journal'),
                        "primary_category": "Pharmacology/Chemistry",
                        "source_db": "PubChem/PubMed",
                        "compound_cid": cid,
                        "compound_name": compound_name
                    }
                    
                    literature_results.append(paper_entry)
            
            # Rate limiting
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error fetching details for PMID {pmid}: {e}")
            continue
    
    return literature_results