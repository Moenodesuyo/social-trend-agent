# Social Trend Agent

Turns a pile of short-form videos (TikTok / Instagram Reels) from one niche into
**marketing language insight** for a brand: which hooks, formats, vocabulary and
emotional tones the niche actually uses, plus concrete post ideas that reuse that
language for the brand's own products. It can also audit the brand's own posts
against those patterns.

The default niche is job hunting / careers. The brand is described in a JSON file,
so you can swap in any brand without changing code. The repo ships with a fictional
example, `brand.example.json`.

```
ingest → fuse text → embed → cluster
       → relevance gate (on-niche clusters only, no API)
       → language insight per cluster (1 batched Gemini call, or a mechanical fallback)
       → marketer-readable report (terminal / JSON / Markdown / Streamlit UI)
```

For each on-niche cluster, the insight contains:

- **theme**
- **hook patterns**, each a template plus a real example line
- **video formats**
- **vocabulary / slang**
- **emotional tone**
- **two brand post ideas** (`brand_application`: tool + ready-to-film hook + angle)
- a **caveat**

## Honesty by design

The output never claims something is "trending" or "going viral". Each insight
carries a caveat that matches the data:

- **No engagement data** (some sources) — it only says a phrasing is *being used*
  in the niche.
- **Engagement present** (sample data, Apify scrapes) — phrasings can be compared by
  views *within this sample*. Even then it is a hashtag-search snapshot with
  cumulative, non-follower-normalized counts, not a time series.

Answering "which phrasing wins over time?" would need repeated collection and
velocity data, which is out of scope for now.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then fill in GEMINI_API_KEY (and APIFY_API_TOKEN if you use Apify)
```

To use your own brand, copy `brand.example.json` to `data/brand.json` and edit it
(name, description, target audience, tone, tools, keywords). If `data/brand.json`
doesn't exist, the example is used. You can also pass `--brand` or set `BRAND_PATH`.
`data/` is gitignored, so your brand profile, scraped data and run history stay local.

## Run

### Streamlit app (recommended)

```bash
streamlit run app.py
```

The UI is bilingual (Japanese / English) and has two tabs:

1. **① Study the niche** — run the pipeline on a data source and read or download
   the insight report.
2. **② Audit your posts** — score the brand's own clips against a saved niche
   reference (see below).

Clustering settings are hidden under "Advanced settings", so a first-time user can
just press **Run analysis**.

### CLI

```bash
# Fully offline: bundled sample data, TF-IDF embeddings, no Gemini
python pipeline.py --backend tfidf --no-gemini

# Semantic embeddings (downloads a small model on first run)
python pipeline.py --backend sentence-transformers

# Apify scraper exports you dropped into data/apify/
python pipeline.py --source apify

# Scrape live via the Apify API (needs APIFY_API_TOKEN)
python pipeline.py --source apify_api --hashtags jobsearch,jobhunt
python pipeline.py --source apify_api --platform instagram --reels-only --hashtags careertok

# Actually watch the top-5 in-scope clips instead of reading captions only
python pipeline.py --source apify_api --platform instagram --hashtags careertok --deep-video-limit 5

# Other useful flags
python pipeline.py --language en              # output language (ja / en)
python pipeline.py --relevance-threshold 80   # send fewer clusters to Gemini
python pipeline.py --clusters 5
python pipeline.py --out out/                 # where to write report.json / report.md
python pipeline.py --no-save --no-reference   # don't write files or a reference snapshot
```

Run `python pipeline.py --help` for the full list.

Each run prints the report and writes `report.json` (structured, for downstream
use) and `report.md` (human-readable). It also saves a timestamped reference
snapshot for the self-audit. The report includes a **sound section**: the share of
the niche using original audio vs. reused tracks, and any sound reused across
several creators.

### With and without a Gemini key

- **With `GEMINI_API_KEY`** — one batched call produces the full insight for *all*
  in-scope clusters, so a run costs one API call. The call uses a strict JSON
  response schema, and a hard timeout (`GEMINI_TIMEOUT`, default 90s) means a hung
  request falls back to local output instead of stalling.
- **Without a key** — the pipeline still runs end to end. `theme` and `vocabulary`
  come from frequency counts, and the model-only fields are marked as needing Gemini.

## Self-audit (② Audit your posts)

You can get the brand's own clips in two ways:

- **Upload `.mp4` files** — each clip is read for its spoken hook, transcript,
  on-screen text and format.
  - By default this runs **locally and for free** (`video_local.py`: faster-whisper +
    EasyOCR + PySceneDetect). The format is a cut-count heuristic, and no visual tags
    are produced.
  - A toggle switches to Gemini's File API instead. That costs one call per clip and
    adds visual tags and a judged format.
  - Paste each clip's post URL and, with an Apify token, the real views / likes /
    comments / saves are fetched and merged in.
- **Apify API** — enter a profile handle or post URLs to pull every post with real
  engagement. Optionally, it also downloads and watches the top N.

The clips are then compared against a saved niche reference in **one batched
Gemini call**. For each clip you get:

- an assessment
- what already works
- gaps
- 1–3 hook rewrites
- a format suggestion
- vocabulary to adopt
- a priority

When real engagement is available, priority takes into account under- and
over-performers *relative to the brand's other clips*.

Other details:

- **Scoreboard** — when several clips are audited, a summary comes first: clip count,
  high-priority count, original-audio share, missing vocabulary shared across clips,
  and cross-cutting recommendations.
- **Reference history** — every niche run saves a new snapshot to
  `data/reference_history/`, keeping the newest `REFERENCE_HISTORY_KEEP` (default 3).
  You choose which one to audit against, and snapshots that are weak (built without
  a key) or in another language are flagged.
- **Audit history** — every audit is saved to `data/audit_history/` (newest
  `AUDIT_HISTORY_KEEP`, default 10) and can be reopened without spending API calls.
- **Downloads** — `self_audit.json` / `.md` / `.pdf`. The PDF uses reportlab's
  built-in Japanese CID font, so no font file is bundled.
- **No key** — video reading still works locally. Scoring falls back to a vocabulary
  set-difference against the reference, plus the reference hook templates.

## Files

| File | Role |
|------|------|
| `pipeline.py` | Orchestrates the flow and prints the report (CLI entry point). |
| `app.py` | Bilingual Streamlit UI (theme in `.streamlit/config.toml`). |
| `config.py` | All settings; reads `.env`. |
| `ingest.py` | The canonical `Video` schema and all loaders: sample JSON, Hugging Face (`lingbow/tiktok-video-engagement-200k`), Apify exports, and live Apify. **The only module that knows where data comes from.** Also has an English-only filter. |
| `apify_client.py` | Minimal Apify client. Starts an actor run, polls until it finishes, then fetches the dataset, so large scrapes aren't limited by Apify's ~300s sync limit. |
| `analyze.py` | Pluggable embeddings (`sentence-transformers` / `tfidf`) + KMeans. |
| `explain.py` | Keyword-based cluster labels. |
| `brand_match.py` | Loads the brand profile; `gatekeeper_score()` is the no-API relevance gate. |
| `insight.py` | The main output: batched Gemini insight extraction, mechanical fallback, caveats. |
| `sound_trends.py` | Original vs. reused audio share, and sounds shared across creators (no API). |
| `video_local.py` | Free local video reading (speech, on-screen text, cuts/format). |
| `video_audit.py` | Self-audit: reads clips (local or Gemini), merges Apify engagement, scores clips against the reference, builds the summary. |
| `exporter.py` | Report / reference / audit I/O, Markdown and PDF export. |
| `sample_videos.json` | 25 synthetic TikTok-style videos (fictional creators and URLs). |
| `brand.example.json` | Fictional example brand profile. |
| `gemini_batch.py`, `generate.py`, `transcribe.py` | Earlier-stage code (old per-cluster scoring/generation, Whisper stub). No longer called by the pipeline; kept for reference. |

## Design choices

- **Swappable ingestion.** Data collection, along with its legality, cost and choice
  of provider, lives only in `ingest.py`. Everything else consumes the neutral
  `Video` shape.
- **Swappable brand.** The brand identity is a JSON file, so a new brand means new
  data, not new code.
- **Swappable embeddings.** `tfidf` needs no downloads or network, so the logic can be
  tested anywhere. `sentence-transformers` gives better semantic grouping.
- **Quota-aware.** The relevance gate is free, and all clusters share one Gemini
  call. Video reading defaults to local.
- **Graceful degradation.** Without a key or a network connection the pipeline still
  produces output, so you can check that everything works before spending on APIs.

## Possible next steps

- Repeated collection + velocity → real trend detection (not just snapshots).
- A genre-aware format classifier for the local video path (today it counts cuts).
- A longer audit history, to see which *older* own posts performed well.

## Data & terms

Scraped data is never committed. If you collect data with Apify or another
scraper, make sure you follow the platforms' terms of service and applicable privacy
law, and keep scraped personal data out of version control.
