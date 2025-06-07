import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from urllib.parse import unquote, parse_qs, urlparse
from typing import List, Dict, Optional, Any
try:
    from config import APP_CONFIG
except ImportError:
    print("Warning: Could not import APP_CONFIG from config.py. TOR proxy settings might not be available.")
    class DummyConfig: 
        TOR_PROXY_IP = "127.0.0.1"
        TOR_PROXY_PORT = "9050" 
    APP_CONFIG = DummyConfig()

def get_tor_proxies() -> Optional[Dict[str, str]]:
    """
    Returns a dictionary with TOR SOCKS proxy configuration.
    Assumes TOR SOCKS proxy is running on 127.0.0.1:9050 or as configured.
    """
    # Attempt to import APP_CONFIG for proxy settings
    try:
        from config import APP_CONFIG
        tor_ip = getattr(APP_CONFIG, "TOR_PROXY_IP", "127.0.0.1")
        tor_port = getattr(APP_CONFIG, "TOR_PROXY_PORT", "9050") # Default SOCKS port for Tor Browser
    except ImportError:
        print("Warning: Could not import APP_CONFIG. Using default TOR proxy settings.")
        tor_ip = "127.0.0.1"
        tor_port = "9050"


    if tor_ip and tor_port:
        return {
            'http': f'socks5h://{tor_ip}:{tor_port}', # socks5h for DNS resolution through Tor
            'https': f'socks5h://{tor_ip}:{tor_port}'
        }
    print("Warning: TOR_PROXY_IP or TOR_PROXY_PORT not configured. TOR proxy will not be used.")
    return None

def search_duckduckgo(query: str, max_results: int = 5, region: str = 'wt-wt') -> List[Dict[str, str]]:
    """
    Performs a web search using DuckDuckGo. (Standard, non-TOR)
    """
    results_list: List[Dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                keywords=query,
                region=region,
                safesearch='moderate',
                max_results=max_results
            )
            if results:
                for res in results: # results is already a list of dicts
                    results_list.append({
                        "title": res.get("title", "N/A"),
                        "href": res.get("href", "#"),
                        "body": res.get("body", "N/A")
                    })
        return results_list
    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")
        return []


def search_duckduckgo_tor(query: str, max_results: int = 7, region: str = 'wt-wt') -> List[Dict[str, str]]:
    """
    Performs a web search using DuckDuckGo's HTML interface through a TOR proxy.
    Parses the HTML to extract results, direct links, and snippets.

    Args:
        query: The search query string.
        max_results: The maximum number of results to attempt to parse.
        region: Region for search (less impactful on HTML version).

    Returns:
        A list of search result dictionaries, or an empty list on error.
    """
    proxies = get_tor_proxies()
    if not proxies:
        print("TOR proxy not configured. Cannot perform search via TOR.")
        return []

    # DuckDuckGo HTML search URL
    search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}&kl={region}"
    headers = { # Using a common user agent
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    }
    
    results_list: List[Dict[str, str]] = []
    try:
        print(f"Searching '{query}' via TOR proxy: {proxies['http']} on URL: {search_url}")
        response = requests.get(search_url, headers=headers, proxies=proxies, timeout=45) # Increased timeout for TOR
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml') # Use lxml for better parsing
        
        # Find result containers. Selector might need adjustment if DDG changes HTML.
        # Common selectors: div.result, div.web-result, li.result-item
        result_blocks = soup.find_all('div', class_='result', limit=max_results * 2) # Get a bit more to filter
        if not result_blocks: # Fallback selector
            result_blocks = soup.find_all('div', class_='web-result', limit=max_results * 2)
        if not result_blocks: # Another fallback
            result_blocks = soup.find_all('li', class_='result-item', limit=max_results * 2)


        count = 0
        for block in result_blocks:
            if count >= max_results:
                break

            title_tag = block.find('a', class_='result__a') or \
                        block.find('a', class_='result-link') or \
                        block.find('h2', class_='result__title') # In case title is in h2
            
            snippet_tag = block.find('a', class_='result__snippet') or \
                          block.find('div', class_='result__snippet') or \
                          block.find('p', class_='result-snippet')


            if title_tag:
                title = title_tag.get_text(strip=True)
                raw_href = title_tag.get('href')
                
                # Extract direct URL from DDG's redirect link (uddg parameter)
                actual_href = raw_href
                if raw_href and 'duckduckgo.com/l/' in raw_href:
                    parsed_url = urlparse(raw_href)
                    query_params = parse_qs(parsed_url.query)
                    if 'uddg' in query_params and query_params['uddg']:
                        actual_href = unquote(query_params['uddg'][0])
                    else: # if no uddg, try to use the raw href if it looks like a full URL
                        actual_href = "https://duckduckgo.com" + raw_href if raw_href.startswith("/") else raw_href
                elif raw_href and not raw_href.startswith("http"): # If relative link not from DDG redirect
                     actual_href = "https://html.duckduckgo.com" + raw_href # Unlikely but handle

                snippet = snippet_tag.get_text(strip=True) if snippet_tag else "Snippet not available."

                if title and actual_href and actual_href.startswith("http"): # Ensure it's a valid-looking URL
                    results_list.append({
                        "title": title,
                        "href": actual_href,
                        "body": snippet
                    })
                    count += 1
            
        if not results_list and result_blocks: # If blocks were found but parsing failed
            print("Found result blocks but failed to parse titles/links. DDG HTML structure might have changed.")
        elif not result_blocks:
            print(f"No result blocks found in HTML for query '{query}'. Page content: {soup.get_text()[:500]}")

        return results_list

    except requests.exceptions.Timeout:
        print(f"Timeout during DuckDuckGo TOR search for '{query}'. TOR might be slow or site unresponsive.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Error during DuckDuckGo TOR search (request failed): {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during DuckDuckGo TOR search: {e}")
        # import traceback
        # traceback.print_exc() # For detailed debugging
        return []

if __name__ == '__main__':
    print("Testing Web Searcher...")
    test_query = "scandium properties"

    print(f"\n--- Standard DuckDuckGo Search for '{test_query}' ---")
    standard_results = search_duckduckgo(test_query, max_results=3)
    if standard_results:
        for i, res in enumerate(standard_results):
            print(f"{i+1}. Title: {res['title']}\n   Link: {res['href']}\n   Snippet: {res['body'][:100]}...\n")
    else:
        print("No results from standard search or error occurred.")

    print(f"\n--- DuckDuckGo Search via TOR for '{test_query}' ---")
    print("Ensure TOR SOCKS proxy is running on 127.0.0.1:9050 (or as configured)")
    tor_results = search_duckduckgo_tor(test_query, max_results=3)
    if tor_results:
        for i, res in enumerate(tor_results):
            print(f"{i+1}. Title: {res['title']}\n   Link: {res['href']}\n   Snippet: {res['body'][:100]}...\n")
    else:
        print("No results from TOR search or error occurred. Check TOR connection and DDG HTML structure.")