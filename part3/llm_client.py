"""
Task 2 — Reusable API wrapper with retry-on-failure handling.

Uses Google Gemini (free API tier, no payment method required — see
README for how to get a free key from Google AI Studio).

The API key is loaded from the GEMINI_API_KEY environment variable via
python-dotenv. It is never hardcoded here.
"""
import os
import time
import json
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current folder and loads it into os.environ

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.0-flash"
BASE_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL_NAME}:generateContent"
)


def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 512,
             max_retries: int = 3) -> str:
    """
    Sends `prompt` to the Gemini API and returns the model's text response.

    Retries up to `max_retries` times (with a short backoff) on network
    errors, rate limits (HTTP 429), or non-200 responses, rather than
    crashing the whole run. If every attempt fails, logs a descriptive
    error and returns an empty string so the caller can skip this record
    and continue.
    """
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create a .env file with "
            "GEMINI_API_KEY=your_key_here (see README)."
        )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    headers = {"Content-Type": "application/json"}
    url = f"{BASE_URL}?key={API_KEY}"

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

            elif response.status_code == 429:
                last_error = f"Rate limited (HTTP 429): {response.text[:200]}"
                print(f"  [retry {attempt}/{max_retries}] {last_error} — backing off...")
                time.sleep(2 ** attempt)  # exponential backoff: 2s, 4s, 8s

            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                print(f"  [retry {attempt}/{max_retries}] {last_error}")
                time.sleep(1)

        except requests.exceptions.RequestException as e:
            last_error = f"Network error: {e}"
            print(f"  [retry {attempt}/{max_retries}] {last_error}")
            time.sleep(1)

        except Exception as e:
            # Catches lower-level failures that don't come wrapped as a
            # requests exception — e.g. SSL certificate verification
            # errors, which on some Windows machines (often due to
            # antivirus/corporate network SSL inspection) raise a raw
            # ssl.SSLCertVerificationError instead of a requests error.
            # Without this, such an error crashes the whole run instead
            # of being retried and logged, which defeats the point of
            # having retry logic at all.
            last_error = f"Unexpected error: {type(e).__name__}: {e}"
            print(f"  [retry {attempt}/{max_retries}] {last_error}")
            time.sleep(1)

    print(f"  [FAILED after {max_retries} attempts] {last_error}")
    return ""


def parse_json_response(raw_text: str):
    """
    Attempts to parse a model response as JSON. Models sometimes wrap JSON
    in markdown code fences (```json ... ```) even when told not to, so
    this strips those before parsing. Returns None (not an exception) if
    parsing fails, so callers can log-and-continue instead of crashing.
    """
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
