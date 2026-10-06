"""
Central configuration for the Social Trend Intelligence Agent (Stage 1a).

All tunable knobs live here so the rest of the code stays clean.
Reads secrets (Gemini API key) from a .env file if present.
"""
import os

# --- Load .env if python-dotenv is available (optional convenience) -----------
# override=True: the .env file is the source of truth. Without it, a stale
# GEMINI_API_KEY exported in the shell profile would silently shadow a key you
# just edited into .env. For a one-off offline run, use `--no-gemini` (or the
# UI's "Gemini APIを使う" toggle) instead of a `GEMINI_API_KEY=""` prefix.
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    # dotenv is optional. If it's missing, we just read real env vars.
    pass

# --- Data source --------------------------------------------------------------
# Stage 1a uses a static local JSON file. Swap this out later for a real
# ingestion connector (API / provider) without touching the rest of the code.
DATA_PATH = os.getenv("DATA_PATH", "sample_videos.json")
# Directory (or single file) of Apify TikTok Scraper JSON, for --source apify.
APIFY_PATH = os.getenv("APIFY_PATH", "data/apify")

# --- Apify API (fetch fresh scrapes from code instead of the web console) ----
# Personal API token from https://console.apify.com/account/integrations .
# Leave empty to keep using downloaded JSON files (APIFY_PATH) only.
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
# Actor ids (owner/name or owner~name). Overridable if you prefer a different
# scraper (e.g. clockworks/free-tiktok-scraper).
APIFY_TIKTOK_ACTOR = os.getenv("APIFY_TIKTOK_ACTOR", "clockworks/tiktok-scraper")
APIFY_INSTAGRAM_ACTOR = os.getenv("APIFY_INSTAGRAM_ACTOR", "apify/instagram-scraper")
# Default cap on rows per API fetch, so a wide hashtag can't burn the free tier.
APIFY_RESULTS_LIMIT = int(os.getenv("APIFY_RESULTS_LIMIT", "100"))
# Max wall-clock time to wait (polling) for an Apify run to finish. Not a
# server-side cap - apify_client.py polls the run's status instead of using
# the run-sync endpoint, so this is purely our own patience and can be raised
# freely for bigger multi-hashtag scrapes.
APIFY_SYNC_TIMEOUT = int(os.getenv("APIFY_SYNC_TIMEOUT", "600"))

# --- Reference history -------------------------------------------------------
# The self-audit ("自己診断") tab scores the brand's own clips against a
# saved niche-insight report. Every niche run writes a NEW timestamped snapshot
# here (instead of overwriting one file) and only the newest N are kept; the
# audit tab lets you pick which snapshot to use.
REFERENCE_HISTORY_DIR = os.getenv("REFERENCE_HISTORY_DIR", "data/reference_history")
REFERENCE_HISTORY_KEEP = int(os.getenv("REFERENCE_HISTORY_KEEP", "3"))

# --- Self-audit history -------------------------------------------------------
# Every completed self-audit run (own clips scored against a niche reference)
# writes a NEW timestamped snapshot here, same shape as the reference history
# above, so a past run can be reopened for free (no Apify / Gemini call) - the
# audit tab lets you pick a saved run instead of always re-running.
AUDIT_HISTORY_DIR = os.getenv("AUDIT_HISTORY_DIR", "data/audit_history")
AUDIT_HISTORY_KEEP = int(os.getenv("AUDIT_HISTORY_KEEP", "10"))

# --- Embedding backend --------------------------------------------------------
# "sentence-transformers"  -> high quality semantic embeddings (recommended;
#                             downloads a model the first time, needs internet once)
# "tfidf"                  -> lightweight sklearn fallback, fully offline,
#                             no download. Good for quick tests.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "all-MiniLM-L6-v2")

# --- Clustering ---------------------------------------------------------------
# Number of clusters (themes) to find. With ~25 videos, 5-7 is sensible.
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "6"))
RANDOM_STATE = 42

# --- Explanation (Gemini) -----------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# Hard wall-clock timeout (seconds) for a single Gemini request, so a hung
# connection can't stall the whole run. On timeout the pipeline falls back to
# the local mechanical insight, exactly like a quota / network error.
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "90"))

# --- Self-audit: analysing the brand's OWN uploaded videos -----------------
# The clip is sent straight to Gemini (File API), so this needs a multimodal
# model and a longer wall-clock budget than a text call (upload + processing +
# generation).
GEMINI_VIDEO_MODEL = os.getenv("GEMINI_VIDEO_MODEL", GEMINI_MODEL)
GEMINI_VIDEO_TIMEOUT = int(os.getenv("GEMINI_VIDEO_TIMEOUT", "300"))

# --- Relevance gate (which clusters get a language insight) -------------------
# Falls back to the fictional example brand when no private data/brand.json exists.
BRAND_PATH = os.getenv("BRAND_PATH") or (
    "data/brand.json" if os.path.exists("data/brand.json") else "brand.example.json"
)
# A cluster gets a language insight only if its gatekeeper score (0-100, share
# of videos hitting a brand keyword) is >= this...
RELEVANCE_THRESHOLD = int(os.getenv("RELEVANCE_THRESHOLD", "60"))
# ...AND it hits at least this many DISTINCT brand keywords, so one accidental
# hashtag collision (e.g. #ats = American Truck Simulator) can't let an
# off-topic cluster through.
MIN_DISTINCT_KEYWORDS = int(os.getenv("MIN_DISTINCT_KEYWORDS", "2"))
