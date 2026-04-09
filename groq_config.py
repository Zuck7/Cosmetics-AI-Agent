import os
from functools import lru_cache

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

try:
    from groq import Groq
except Exception:
    Groq = None


@lru_cache(maxsize=1)
def get_groq_api_key():
    """Return the single Groq API key for the project from .env."""
    # Enforce .env as the source of truth for this process.
    load_dotenv(override=True)
    key = os.getenv("GROQ_API_KEY", "").strip()
    return key or None


@lru_cache(maxsize=1)
def get_groq_client():
    """Build a shared Groq client using the project key."""
    if Groq is None:
        return None

    api_key = get_groq_api_key()
    if not api_key:
        return None

    try:
        return Groq(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize Groq client: {e}. Falling back to demo mode.")
        return None
