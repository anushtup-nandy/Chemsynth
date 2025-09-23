# utils/chemrxiv_processor.py
import requests
from typing import List, Dict, Any, Optional
import time
from urllib.parse import quote
from bs4 import BeautifulSoup
import json

# ChemRxiv API endpoints
CHEMRXIV_API_BASE = "https://chemrxiv.org/engage/chemrxiv/public-api/v1"
CHEMRXIV_SEARCH_URL = f"{CHEMRXIV_API_BASE}/items"
CHEMRXIV_ITEM_URL = f"{CHEMRXIV_API_BASE}/items"

def search_chemrxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches ChemRxiv for preprints matching the query.
    
    Args:
        query: The search query (chemical name, compound, synthesis terms, etc.)
        max_results: Maximum number of results to return
        
    Returns:
        A list of dictionaries containing preprint metadata
    """
    print(f"Searching ChemRxiv for query: '{query}'")
    results = []
    
    # Fixed ChemRxiv search parameters with correct sort value
    params = {
        'term': query,
        'limit': min(max_results * 2, 50),  # Fetch extra to allow for filtering
        'skip': 0,
        'sort': 'RELEVANT_DESC'  # Fixed: was 'RELEVANCE', now using correct API value
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    try:
        print(f"Making request to ChemRxiv API with params: {params}")
        response = requests.get(CHEMRXIV_SEARCH_URL, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('itemHits', [])
            
            if not items:
                print("No results found in ChemRxiv search")
                return []
                
            print(f"Found {len(items)} items from ChemRxiv")
            
            for item in items[:max_results]:
                try:
                    paper_data = _extract_chemrxiv_metadata(item)
                    if paper_data:
                        results.append(paper_data)
                except Exception as e:
                    print(f"Error processing ChemRxiv item: {e}")
                    continue
                    
        elif response.status_code == 429:
            print("Rate limited by ChemRxiv API, waiting and retrying...")
            time.sleep(2)
            return _retry_chemrxiv_search(query, max_results, headers)
        else:
            print(f"ChemRxiv API returned status {response.status_code}: {response.text}")
            # Fallback to web scraping if API fails
            return _fallback_web_scrape_chemrxiv(query, max_results)
            
    except requests.exceptions.RequestException as e:
        print(f"Request error during ChemRxiv search: {e}")
        # Fallback to web scraping
        return _fallback_web_scrape_chemrxiv(query, max_results)
    except Exception as e:
        print(f"Unexpected error during ChemRxiv search: {e}")
        return []
    
    return results

def _extract_chemrxiv_metadata(item: Dict) -> Optional[Dict[str, Any]]:
    """
    Extracts metadata from a ChemRxiv API response item.
    Enhanced error handling and data extraction.
    """
    try:
        # Get the main item data - handle both nested and flat structures
        item_data = item.get('item', item)  # Fallback to item itself if no nested 'item'
        if not item_data:
            return None
            
        # Extract authors with better handling
        authors = []
        author_list = item_data.get('authors', [])
        for author in author_list:
            if isinstance(author, dict):
                first_name = author.get('firstName', '').strip()
                last_name = author.get('lastName', '').strip()
                if first_name and last_name:
                    authors.append(f"{first_name} {last_name}")
                elif last_name:
                    authors.append(last_name)
                elif first_name:
                    authors.append(first_name)
            elif isinstance(author, str) and author.strip():
                authors.append(author.strip())
        
        # Extract publication dates with fallbacks
        published_date = (item_data.get('publishedDate') or 
                         item_data.get('createdDate') or 
                         item_data.get('submittedDate') or 'Unknown')
        created_date = item_data.get('createdDate', '')
        
        # Extract DOI and URLs
        doi = item_data.get('doi', '')
        item_id = item_data.get('id', '') or item_data.get('itemId', '')
        
        # Construct URLs with better logic
        pdf_url = ""
        if doi:
            pdf_url = f"https://doi.org/{doi}"
        elif item_id:
            pdf_url = f"https://chemrxiv.org/engage/chemrxiv/article-details/{item_id}"
        
        # Extract categories/subjects with better handling
        categories = []
        subjects = item_data.get('categories', []) or item_data.get('subjects', [])
        for subject in subjects:
            if isinstance(subject, dict):
                name = subject.get('name', '') or subject.get('label', '')
                if name:
                    categories.append(name)
            elif isinstance(subject, str) and subject.strip():
                categories.append(subject.strip())
        
        # Default category if none found
        if not categories:
            categories = ["Chemistry"]
        
        # Get version info
        version = item_data.get('version', 1)
        
        # Extract metrics safely
        metrics = item_data.get('metrics', {})
        views = metrics.get('views', 0) if isinstance(metrics, dict) else 0
        downloads = metrics.get('downloads', 0) if isinstance(metrics, dict) else 0
        
        # Get title and abstract with fallbacks
        title = item_data.get('title', '').strip() or 'No title available'
        abstract = (item_data.get('abstract', '') or 
                   item_data.get('description', '') or 
                   'No abstract available').strip()
        
        paper_entry = {
            "entry_id": f"ChemRxiv:{item_id}" if item_id else f"ChemRxiv:unknown_{hash(title)}",
            "title": title,
            "authors": authors or ["Unknown Author"],
            "summary": abstract,
            "published": published_date,
            "updated": item_data.get('modifiedDate', '') or item_data.get('updatedDate', ''),
            "categories": categories,
            "doi": doi,
            "pdf_url": pdf_url,
            "comment": f"ChemRxiv preprint, Version {version}",
            "journal_ref": f"ChemRxiv preprint (Version {version})",
            "primary_category": categories[0] if categories else "Chemistry",
            "source_db": "ChemRxiv",
            "version": version,
            "views": views,
            "downloads": downloads
        }
        
        return paper_entry
        
    except Exception as e:
        print(f"Error extracting ChemRxiv metadata: {e}")
        return None

def _retry_chemrxiv_search(query: str, max_results: int, headers: Dict, max_retries: int = 2) -> List[Dict[str, Any]]:
    """
    Retry ChemRxiv search with exponential backoff.
    """
    for attempt in range(max_retries):
        wait_time = 2 ** (attempt + 1)
        print(f"Retrying ChemRxiv search in {wait_time} seconds (attempt {attempt + 1}/{max_retries})")
        time.sleep(wait_time)
        
        try:
            params = {
                'term': query,
                'limit': min(max_results * 2, 50),
                'skip': 0,
                'sort': 'RELEVANT_DESC'  # Fixed sort parameter
            }
            
            response = requests.get(CHEMRXIV_SEARCH_URL, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('itemHits', [])
                results = []
                
                for item in items[:max_results]:
                    try:
                        paper_data = _extract_chemrxiv_metadata(item)
                        if paper_data:
                            results.append(paper_data)
                    except Exception as e:
                        print(f"Error processing ChemRxiv item in retry: {e}")
                        continue
                        
                return results
                
        except Exception as e:
            print(f"Retry attempt {attempt + 1} failed: {e}")
            continue
            
    print("All retry attempts failed, falling back to web scraping")
    return _fallback_web_scrape_chemrxiv(query, max_results)

def _fallback_web_scrape_chemrxiv(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Improved fallback method to scrape ChemRxiv search results when API fails.
    """
    print(f"Attempting to scrape ChemRxiv web interface for query: '{query}'")
    results = []
    
    try:
        # Updated ChemRxiv search URL - try the main search endpoint
        search_url = "https://chemrxiv.org/engage/chemrxiv/search-dashboard"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none'
        }
        
        # Try with a session for better cookie handling
        session = requests.Session()
        session.headers.update(headers)
        
        # First, get the main page to establish session
        main_response = session.get("https://chemrxiv.org", timeout=15)
        time.sleep(1)  # Brief delay
        
        # Then search
        search_params = {'q': query}
        response = session.get(search_url, params=search_params, timeout=20)
        
        if response.status_code == 200:
            # Use more flexible parsing with error handling
            try:
                soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
            except UnicodeDecodeError:
                # Try with different encoding
                soup = BeautifulSoup(response.content.decode('utf-8', errors='replace'), 'html.parser')
            
            # Look for various possible containers
            selectors_to_try = [
                'div[class*="search-result"]',
                'div[class*="item-container"]', 
                'div[class*="paper"]',
                'div[class*="preprint"]',
                'article',
                'div[class*="card"]'
            ]
            
            article_containers = []
            for selector in selectors_to_try:
                containers = soup.select(selector)
                if containers:
                    article_containers = containers
                    print(f"Found containers with selector: {selector}")
                    break
            
            count = 0
            for container in article_containers:
                if count >= max_results:
                    break
                    
                try:
                    # Try multiple title selectors
                    title = None
                    for title_selector in ['h1', 'h2', 'h3', 'h4', '[class*="title"]', 'a[href*="article"]']:
                        title_elem = container.select_one(title_selector)
                        if title_elem and title_elem.get_text(strip=True):
                            title = title_elem.get_text(strip=True)
                            break
                    
                    if not title or title == "Title not found":
                        continue
                    
                    # Extract link
                    link = ""
                    link_elem = container.find('a', href=True)
                    if link_elem and link_elem.get('href'):
                        link = link_elem['href']
                        if link and not link.startswith('http'):
                            link = f"https://chemrxiv.org{link}"
                    
                    # Extract authors - be more flexible
                    authors = []
                    author_patterns = [
                        'span[class*="author"]',
                        'div[class*="author"]',
                        'p[class*="author"]'
                    ]
                    
                    for pattern in author_patterns:
                        author_elems = container.select(pattern)
                        if author_elems:
                            for elem in author_elems:
                                author_text = elem.get_text(strip=True)
                                if author_text and len(author_text) > 1:
                                    # Clean up author text
                                    author_text = author_text.replace('Authors:', '').replace('By:', '').strip()
                                    if author_text:
                                        authors.extend([a.strip() for a in author_text.split(',') if a.strip()])
                            break
                    
                    # Extract abstract/summary
                    abstract = "Abstract not available"
                    abstract_selectors = [
                        'div[class*="abstract"]',
                        'p[class*="abstract"]', 
                        'div[class*="summary"]',
                        'p'  # Fallback to first paragraph
                    ]
                    
                    for selector in abstract_selectors:
                        abstract_elem = container.select_one(selector)
                        if abstract_elem:
                            text = abstract_elem.get_text(strip=True)
                            if text and len(text) > 20:  # Make sure it's substantial
                                abstract = text[:500]  # Truncate if too long
                                break
                    
                    paper_entry = {
                        "entry_id": f"ChemRxiv:scraped:{count}",
                        "title": title,
                        "authors": authors or ["Unknown Author"],
                        "summary": abstract,
                        "published": "Unknown",
                        "updated": "",
                        "categories": ["Chemistry"],
                        "doi": "",
                        "pdf_url": link,
                        "comment": "ChemRxiv preprint (scraped)",
                        "journal_ref": "ChemRxiv preprint",
                        "primary_category": "Chemistry",
                        "source_db": "ChemRxiv (scraped)"
                    }
                    results.append(paper_entry)
                    count += 1
                    
                except Exception as e:
                    print(f"Error extracting data from scraped container: {e}")
                    continue
                    
        else:
            print(f"Failed to scrape ChemRxiv: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"Error during ChemRxiv web scraping: {e}")
    
    print(f"Scraped {len(results)} results from ChemRxiv")
    return results

def get_chemrxiv_item_details(item_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch detailed information about a specific ChemRxiv item.
    """
    try:
        url = f"{CHEMRXIV_ITEM_URL}/{item_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch ChemRxiv item {item_id}: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching ChemRxiv item details for {item_id}: {e}")
        return None

def search_chemrxiv_advanced(
    query: str, 
    max_results: int = 5,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort_by: str = 'RELEVANT_DESC'  # Fixed default value
) -> List[Dict[str, Any]]:
    """
    Advanced ChemRxiv search with additional filtering options.
    
    Args:
        query: Search query
        max_results: Maximum number of results
        date_from: Start date in YYYY-MM-DD format
        date_to: End date in YYYY-MM-DD format  
        sort_by: Sorting criterion - valid options:
                VIEWS_COUNT_ASC, VIEWS_COUNT_DESC, CITATION_COUNT_ASC, 
                CITATION_COUNT_DESC, READ_COUNT_ASC, READ_COUNT_DESC,
                RELEVANT_ASC, RELEVANT_DESC, PUBLISHED_DATE_ASC, PUBLISHED_DATE_DESC
        
    Returns:
        List of paper metadata dictionaries
    """
    # Validate sort_by parameter
    valid_sorts = [
        'VIEWS_COUNT_ASC', 'VIEWS_COUNT_DESC', 'CITATION_COUNT_ASC', 
        'CITATION_COUNT_DESC', 'READ_COUNT_ASC', 'READ_COUNT_DESC',
        'RELEVANT_ASC', 'RELEVANT_DESC', 'PUBLISHED_DATE_ASC', 'PUBLISHED_DATE_DESC'
    ]
    
    if sort_by not in valid_sorts:
        print(f"Invalid sort_by value: {sort_by}. Using RELEVANT_DESC instead.")
        sort_by = 'RELEVANT_DESC'
    
    params = {
        'term': query,
        'limit': min(max_results * 2, 50),
        'skip': 0,
        'sort': sort_by
    }
    
    # Add date filters if provided
    if date_from:
        params['dateFrom'] = date_from
    if date_to:
        params['dateTo'] = date_to
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    try:
        print(f"Advanced search with params: {params}")
        response = requests.get(CHEMRXIV_SEARCH_URL, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('itemHits', [])
            results = []
            
            print(f"Advanced search found {len(items)} items")
            
            for item in items[:max_results]:
                try:
                    paper_data = _extract_chemrxiv_metadata(item)
                    if paper_data:
                        results.append(paper_data)
                except Exception as e:
                    print(f"Error processing advanced search item: {e}")
                    continue
                    
            return results
        else:
            print(f"Advanced ChemRxiv search failed: HTTP {response.status_code} - {response.text}")
            return search_chemrxiv(query, max_results)  # Fallback to basic search
            
    except Exception as e:
        print(f"Error in advanced ChemRxiv search: {e}")
        return search_chemrxiv(query, max_results)  # Fallback to basic search

if __name__ == '__main__':
    # Test the ChemRxiv search functionality
    test_queries = [
        "ibuprofen synthesis",
        "organic photocatalysis", 
        "drug discovery",
        "sustainable chemistry"
    ]
    
    for query in test_queries:
        print(f"\n=== Testing ChemRxiv search for: '{query}' ===")
        results = search_chemrxiv(query, max_results=3)
        
        if results:
            for i, paper in enumerate(results, 1):
                print(f"\n{i}. {paper['title']}")
                print(f"   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
                print(f"   Published: {paper['published']}")
                print(f"   URL: {paper['pdf_url']}")
                print(f"   Abstract: {paper['summary'][:150]}...")
        else:
            print(f"No results found for '{query}'")
    
    print("\n=== Testing advanced search ===")
    advanced_results = search_chemrxiv_advanced(
        "catalysis", 
        max_results=2, 
        sort_by='PUBLISHED_DATE_DESC'  # Fixed sort value
    )
    
    if advanced_results:
        for paper in advanced_results:
            print(f"Advanced result: {paper['title']}")
    else:
        print("No advanced results found")