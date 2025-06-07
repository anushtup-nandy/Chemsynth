# utils/__init__.py

from .llm_interface import generate_text
from .web_searcher import search_duckduckgo, search_duckduckgo_tor
from .web_scraper import fetch_url_content, extract_text_from_html
from .arxiv_processor import search_arxiv, summarize_arxiv_paper_abstract
# Ensure search_github_repos is imported from github_analyzer
from .github_analyzer import (
    get_github_repo_readme,
    summarize_github_readme,
    search_github_repos  # <<< ADD THIS IMPORT
)
from .prompt_loader import get_prompt_template, format_prompt

__all__ = [
    "generate_text",
    "search_duckduckgo",
    "search_duckduckgo_tor",
    "fetch_url_content",
    "extract_text_from_html",
    "search_arxiv",
    "summarize_arxiv_paper_abstract",
    "get_github_repo_readme",
    "summarize_github_readme",
    "search_github_repos",  # <<< AND ADD IT HERE
    "get_prompt_template",
    "format_prompt",
]