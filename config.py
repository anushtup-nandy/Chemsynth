# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TOR_PROXY_IP = os.getenv("TOR_PROXY_IP", "127.0.0.1")
    TOR_PROXY_PORT = os.getenv("TOR_PROXY_PORT", "9050")
    # GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") # Optional: for higher GitHub API rate limits

    # Flask specific settings (optional for now)
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    SECRET_KEY = os.getenv("SECRET_KEY", "a_very_secret_key_for_dev_only")
    EMAIL = os.getenv("EMAIL")


APP_CONFIG = Config()