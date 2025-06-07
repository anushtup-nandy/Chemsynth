# utils/web_scraper.py
import requests
from bs4 import BeautifulSoup, Comment
from typing import Optional, Dict, List
import time

# Attempt to import APP_CONFIG for proxy settings
try:
    from config import APP_CONFIG
except ImportError:
    print("Warning: Could not import APP_CONFIG from config.py. TOR proxy settings might not be available.")
    class DummyConfig:
        TOR_PROXY_IP = "127.0.0.1"
        TOR_PROXY_PORT = "9050"
    APP_CONFIG = DummyConfig()

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

def get_tor_session() -> Optional[requests.Session]:
    """
    Returns a requests.Session configured to use TOR SOCKS proxy.
    """
    tor_ip = getattr(APP_CONFIG, "TOR_PROXY_IP", "127.0.0.1")
    tor_port = getattr(APP_CONFIG, "TOR_PROXY_PORT", "9050")

    if tor_ip and tor_port:
        session = requests.Session()
        session.proxies = {
            'http': f'socks5h://{tor_ip}:{tor_port}',
            'https': f'socks5h://{tor_ip}:{tor_port}'
        }
        return session
    print("Warning: TOR_PROXY_IP or TOR_PROXY_PORT not configured. Cannot create TOR session.")
    return None

def fetch_url_content(
    url: str,
    use_tor: bool = False,
    timeout: int = 20,
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = 2,
    retry_delay: int = 3 # seconds
) -> Optional[str]:
    """
    Fetches the HTML content of a given URL.

    Args:
        url: The URL to fetch.
        use_tor: Whether to route the request through TOR.
        timeout: Request timeout in seconds.
        headers: Custom headers for the request.
        max_retries: Number of times to retry on failure.
        retry_delay: Delay between retries.

    Returns:
        The HTML content as a string, or None if an error occurs.
    """
    effective_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        effective_headers.update(headers)

    session = None
    if use_tor:
        session = get_tor_session()
        if not session:
            print(f"Failed to get TOR session for URL: {url}")
            return None
        print(f"Fetching {url} via TOR proxy...")
    else:
        session = requests.Session()
        print(f"Fetching {url} directly...")
    
    session.headers.update(effective_headers)

    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4XX or 5XX)
            # Ensure content is decoded correctly, try utf-8 if apparent_encoding fails
            content = ""
            if response.content:
                try:
                    content = response.content.decode(response.apparent_encoding or 'utf-8', errors='ignore')
                except Exception: # Fallback if apparent_encoding is problematic
                    content = response.content.decode('utf-8', errors='ignore')
            return content
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error fetching {url} (attempt {attempt+1}/{max_retries}): {e.response.status_code} {e.response.reason}")
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error fetching {url} (attempt {attempt+1}/{max_retries}): {e}")
        except requests.exceptions.Timeout as e:
            print(f"Timeout error fetching {url} (attempt {attempt+1}/{max_retries}): {e}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url} (attempt {attempt+1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (attempt + 1)) # Exponential backoff could be better
        else:
            print(f"Max retries reached for {url}.")
    
    return None


def extract_text_from_html(
    html_content: str,
    unwanted_tags: List[str] = ["script", "style", "nav", "footer", "aside", "header", "form"],
    separator: str = "\n"
) -> str:
    """
    Extracts meaningful text content from HTML using BeautifulSoup.
    Removes scripts, styles, comments, and other specified unwanted tags.

    Args:
        html_content: The HTML content as a string.
        unwanted_tags: A list of HTML tags to remove before text extraction.
        separator: The separator to use between text blocks from different elements.

    Returns:
        The extracted text content as a single string.
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "lxml") # lxml is generally faster and more robust

    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove unwanted tags and their content
    for tag_name in unwanted_tags:
        for tag in soup.find_all(tag_name):
            tag.decompose() # decompose removes the tag and its content

    # Get text, joining with the specified separator
    # .get_text() has a 'separator' argument that can be useful
    text_parts = []
    # Iterate through elements that are more likely to contain main content
    # This is a heuristic, might need adjustment based on common page structures
    main_content_tags = soup.find_all(['p', 'div', 'article', 'main', 'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td'])
    
    if not main_content_tags: # Fallback if specific tags aren't found
        return soup.get_text(separator=separator, strip=True)

    for element in main_content_tags:
        # Avoid extracting text from already processed children if parent is in main_content_tags
        # This simple iteration might still have some redundancy, but get_text(strip=True) on elements helps.
        element_text = element.get_text(separator=" ", strip=True) # Use space for intra-element text
        if element_text:
            text_parts.append(element_text)
            
    return separator.join(text_parts)

if __name__ == '__main__':
    print("Testing Web Scraper...")
    # Test with a known simple static site (or a local HTML file for reliability)
    # Wikipedia is generally good for testing static scraping
    test_url_static = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    
    print(f"\n--- Fetching static content from: {test_url_static} ---")
    html = fetch_url_content(test_url_static)
    if html:
        print(f"Successfully fetched {len(html)} bytes.")
        extracted_text = extract_text_from_html(html)
        print(f"\nExtracted Text (first 500 chars):\n{extracted_text[:500]}...")
        
        # Example of how many words (approx)
        word_count = len(extracted_text.split())
        print(f"\nApproximate word count: {word_count}")
    else:
        print(f"Failed to fetch content from {test_url_static}")

    print("\n--- Testing TOR fetch (if configured and TOR is running) ---")
    # Use a site known to be TOR-friendly, like duckduckgo.com itself, or a hidden service if you have one for testing
    # Be mindful of the site's ToS when scraping.
    test_url_tor = "https://check.torproject.org/" # Good for verifying TOR connection
    # test_url_tor = "http://httpbin.org/ip" # See what IP it reports

    if get_tor_session(): # Check if TOR might be configured
        print(f"Attempting to fetch {test_url_tor} via TOR...")
        tor_html = fetch_url_content(test_url_tor, use_tor=True)
        if tor_html:
            print(f"Successfully fetched {len(tor_html)} bytes via TOR.")
            tor_text = extract_text_from_html(tor_html)
            print(f"\nTOR Extracted Text (first 300 chars):\n{tor_text[:300]}...")
            if "Congratulations. This browser is configured to use Tor." in tor_text:
                 print("\nTOR connection confirmed by check.torproject.org!")
            # If using httpbin.org/ip, parse the JSON to see the origin IP
            # import json
            # try:
            #     ip_data = json.loads(tor_text) # Assuming httpbin.org/ip returns JSON
            #     print(f"IP address reported by httpbin (via TOR): {ip_data.get('origin')}")
            # except:
            #      pass
        else:
            print(f"Failed to fetch content from {test_url_tor} via TOR. Ensure TOR is running and accessible.")
    else:
        print("Skipping TOR fetch test: TOR proxy not configured or session creation failed.")

    print("\n--- Robustness Note ---")
    print("The current scraper is for static HTML. For JavaScript-heavy sites,")
    print("consider using libraries like Playwright or Selenium.")