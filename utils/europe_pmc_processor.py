import requests
from typing import List, Dict, Any

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

def search_europe_pmc(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches Europe PMC for literature.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.

    Returns:
        A list of dictionaries containing paper metadata.
    """
    params = {
        'query': query,
        'resultType': 'core',
        'pageSize': max_results,
        'format': 'json'
    }
    
    literature_results = []
    print(f"Executing Europe PMC search with query: '{query}'")

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        
        data = response.json()
        results = data.get('resultList', {}).get('result', [])

        if not results:
            print("No results found in Europe PMC.")
            return []

        for res in results:
            authors = [author.get('fullName', 'N/A') for author in res.get('authorList', {}).get('author', [])]
            
            paper_entry = {
                "entry_id": f"PMCID:{res.get('pmcid', res.get('id'))}",
                "title": res.get('title', 'No title available'),
                "authors": authors,
                "summary": res.get('abstractText', 'No abstract available'),
                "published": res.get('firstPublicationDate', 'Unknown'),
                "updated": res.get('firstIndexDate', ''),
                "categories": [src.get('source') for src in res.get('meshHeadingList', {}).get('meshHeading', [])],
                "doi": res.get('doi', None),
                "pdf_url": f"https://doi.org/{res.get('doi')}" if res.get('doi') else res.get('fullTextUrlList', {}).get('fullTextUrl', [{}])[0].get('url'),
                "comment": f"PMCID: {res.get('pmcid', 'N/A')}, Citations: {res.get('citedByCount', 0)}",
                "journal_ref": res.get('journalInfo', {}).get('journal', {}).get('title', 'Unknown Journal'),
                "primary_category": res.get('subsetList', {}).get('subset', [{}])[0].get('code', "Biomedical"),
                "source_db": "Europe PMC"
            }
            literature_results.append(paper_entry)
            
        return literature_results

    except requests.exceptions.RequestException as e:
        print(f"Error during Europe PMC search for query '{query}': {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred processing Europe PMC results: {e}")
        return []