import requests
from typing import Optional, Dict, List, Any
import base64

from .llm_interface import generate_text
from .prompt_loader import format_prompt, get_prompt_template

# Attempt to import GITHUB_TOKEN from config
try:
    from config import APP_CONFIG
    GITHUB_TOKEN = getattr(APP_CONFIG, "GITHUB_TOKEN", None)
except ImportError:
    GITHUB_TOKEN = None
    print("Warning: Could not import APP_CONFIG. GITHUB_TOKEN will not be used.")


DEFAULT_GITHUB_API_BASE = "https://api.github.com"

def _get_github_headers() -> Dict[str, str]:
    """Prepares headers for GitHub API requests."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    else:
        print("Note: Making unauthenticated GitHub API requests. Rate limits are lower.")
    return headers

def get_github_repo_readme(
    owner: str,
    repo: str,
    branch: Optional[str] = None # Main/master branch usually, API can find default
) -> Optional[Dict[str, Any]]:
    """
    Fetches the content of a GitHub repository's README file.

    Args:
        owner: The owner of the repository (e.g., "torvalds").
        repo: The name of the repository (e.g., "linux").
        branch: The specific branch to get the README from. If None, API attempts default.

    Returns:
        A dictionary containing 'name', 'path', 'sha', 'size', 'url', 'html_url',
        'git_url', 'download_url', 'type', 'content' (base64 encoded),
        'decoded_content' (UTF-8 decoded string), and '_links', or None on error.
    """
    readme_url = f"{DEFAULT_GITHUB_API_BASE}/repos/{owner}/{repo}/readme"
    if branch:
        readme_url += f"?ref={branch}"
    
    headers = _get_github_headers()
    
    try:
        print(f"Fetching README for {owner}/{repo} from {readme_url}")
        response = requests.get(readme_url, headers=headers, timeout=15)
        response.raise_for_status()
        readme_data = response.json()

        if readme_data.get("content") and readme_data.get("encoding") == "base64":
            try:
                decoded_bytes = base64.b64decode(readme_data["content"])
                readme_data["decoded_content"] = decoded_bytes.decode('utf-8')
            except Exception as decode_err:
                print(f"Error decoding README content for {owner}/{repo}: {decode_err}")
                readme_data["decoded_content"] = None # Or indicate error
        else:
            readme_data["decoded_content"] = None # Content might not be base64 or might be empty

        return readme_data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"README not found for {owner}/{repo} (status 404). URL: {readme_url}")
        else:
            print(f"HTTP error fetching README for {owner}/{repo}: {e.response.status_code} {e.response.reason}. URL: {readme_url}")
    except requests.exceptions.RequestException as e:
        print(f"Request error fetching README for {owner}/{repo}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred fetching README for {owner}/{repo}: {e}")
    return None

def summarize_github_readme(
    readme_content: str, # The decoded README text
    repo_full_name: str, # "owner/repo" for context
    llm_model_name: Optional[str] = None,
    prompt_key: str = "summarize_github_readme",
    prompt_file: str = "task_specific_prompts.yaml"
) -> Optional[str]:
    """
    Summarizes GitHub README content using an LLM.

    Args:
        readme_content: The textual content of the README file.
        repo_full_name: The repository name (e.g., "user/my-project") for context.
        llm_model_name: Optional name of the LLM model to use.
        prompt_key: Key for the summarization prompt.
        prompt_file: YAML file containing the prompt.

    Returns:
        The summarized README text, or None on failure.
    """
    if not readme_content:
        print(f"Error: README content for '{repo_full_name}' is empty. Cannot summarize.")
        return None

    system_persona_data = get_prompt_template("code_documentation_assistant", "system_prompts.yaml")
    system_instruction = system_persona_data.get("code_documentation_assistant") if system_persona_data else None

    formatted_prompt = format_prompt(
        prompt_key,
        prompt_file,
        readme_content=readme_content # Ensure this matches input_variables in YAML
    )

    if not formatted_prompt:
        print(f"Error: Could not format prompt '{prompt_key}' for repo '{repo_full_name}'")
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

def search_github_repos(
    query: str,
    max_results: int = 5,
    sort: str = "stars", # stars, forks, help-wanted-issues, updated
    order: str = "desc"  # asc, desc
) -> Optional[List[Dict[str, Any]]]:
    """
    Searches GitHub for repositories.

    Args:
        query: Search query (e.g., "flask authentication", "language:python topic:nlp").
        max_results: Max number of repositories to return.
        sort: Sorting criterion.
        order: Sorting order.

    Returns:
        A list of simplified repository data dictionaries, or None on error.
    """
    search_url = f"{DEFAULT_GITHUB_API_BASE}/search/repositories"
    params = {
        'q': query,
        'sort': sort,
        'order': order,
        'per_page': max_results
    }
    headers = _get_github_headers()
    repo_list: List[Dict[str, Any]] = []

    try:
        print(f"Searching GitHub repos for: '{query}' with params: {params}")
        response = requests.get(search_url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        search_data = response.json()
        
        if search_data and "items" in search_data:
            for item in search_data["items"]:
                repo_list.append({
                    "full_name": item.get("full_name"),
                    "html_url": item.get("html_url"),
                    "description": item.get("description", "N/A"),
                    "stars": item.get("stargazers_count", 0),
                    "forks": item.get("forks_count", 0),
                    "language": item.get("language", "N/A"),
                    "owner_login": item.get("owner", {}).get("login") if item.get("owner") else "N/A",
                    "repo_name": item.get("name") # for fetching readme later
                })
            return repo_list
        else:
            print(f"No 'items' found in GitHub search response for query: {query}")
            return [] # Return empty list if 'items' key is missing or no results
            
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error searching GitHub repos: {e.response.status_code} {e.response.reason}. URL: {e.request.url}")
        if e.response.status_code == 403:
            print("GitHub API rate limit likely exceeded. Try adding a GITHUB_TOKEN to your .env file or wait.")
    except requests.exceptions.RequestException as e:
        print(f"Request error searching GitHub repos: {e}")
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred while searching GitHub repos: {e}")
    return None # Indicates an error occurred during the process

if __name__ == '__main__':
    print("Testing GitHub Analyzer...")
    # Use a well-known public repository
    test_owner = "pallets"
    test_repo = "flask"
    repo_full_name = f"{test_owner}/{test_repo}"

    print(f"\n--- Fetching README for: {repo_full_name} ---")
    readme_data = get_github_repo_readme(test_owner, test_repo)

    if readme_data and readme_data.get("decoded_content"):
        print(f"Successfully fetched and decoded README for {repo_full_name}.")
        print(f"README Name: {readme_data.get('name')}")
        print(f"README Size: {readme_data.get('size')} bytes")
        readme_text = readme_data["decoded_content"]
        print(f"README Content (first 300 chars):\n{readme_text[:300]}...\n")

        # Test README Summarization
        # Ensure your GEMINI_API_KEY is set in .env for this to work
        print(f"\n--- Summarizing README for {repo_full_name} ---")
        try:
            from config import APP_CONFIG
            if not hasattr(APP_CONFIG, 'GEMINI_API_KEY') or not APP_CONFIG.GEMINI_API_KEY:
                print("Skipping LLM summarization: GEMINI_API_KEY not found in config.")
            else:
                summary = summarize_github_readme(readme_text, repo_full_name)
                if summary:
                    print(f"\nLLM Summary for {repo_full_name} README:\n{summary}")
                else:
                    print(f"Failed to generate summary for {repo_full_name} README.")
        except ImportError:
             print("Skipping LLM summarization: config.py not found (likely standalone run).")
    elif readme_data and not readme_data.get("decoded_content"):
        print(f"Fetched README metadata for {repo_full_name}, but content decoding failed or content was empty.")
    else:
        print(f"Failed to fetch README for {repo_full_name}.")

    print("\n--- Testing GitHub Repo Search ---")
    search_query_gh = "python flask authentication example"
    print(f"Searching for: '{search_query_gh}' (max 3 results, sorted by stars)")
    
    found_repos = search_github_repos(search_query_gh, max_results=3, sort="stars")
    
    if found_repos is not None: # Check for None explicitly, empty list is a valid result
        if found_repos:
            print(f"Found {len(found_repos)} repositories:")
            for i, repo_item in enumerate(found_repos):
                print(f"  {i+1}. {repo_item.get('full_name')} - Stars: {repo_item.get('stars')}")
                print(f"     Desc: {repo_item.get('description', 'N/A')[:100]}...")
                print(f"     URL: {repo_item.get('html_url')}")
                print(f"     Owner: {repo_item.get('owner_login')}, Repo: {repo_item.get('repo_name')}")
        else:
            print("No repositories found matching the criteria.")
    else: # None was returned, indicating an error during the search process
        print("Error occurred during GitHub repository search.")