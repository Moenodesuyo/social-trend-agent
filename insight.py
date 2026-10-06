"""
Language-insight layer (the new deliverable).

Goal shift: this project used to *score* how relevant a trend is to a brand.
Now the deliverable is marketing-facing LANGUAGE INSIGHT for the brand's
short-form social (Instagram Reels / TikTok): for each job-hunt / career
cluster, what hooks, formats, vocabulary and emotional tones are actually
BEING USED - phrased so a marketer can drop them straight into a post.

Honesty guardrail
-----------------
Two data situations, two caveats (see caveat_for()):

* No engagement (the sample JSON, Hugging Face): we can only say a phrasing is
  "being used" in the niche, never that it is "trending" or "blowing up".
* Engagement present (Apify scrape - playCount / diggCount): we may compare
  phrasings/formats WITHIN this sample by view count, but it is a keyword-search
  snapshot, not a time series, and the counts are cumulative and not
  follower-normalised - so still no "it's going viral right now" claims.

Every insight carries the matching `caveat`, and nothing here claims virality.

Primary path: one Gemini call for ALL in-scope clusters at once
(`extract_insights_batch`) so a full run costs a single API call.

Fallback path (no GEMINI_API_KEY / call failed): mechanically tally the
frequent hashtags and words to fill `vocabulary` and `theme`; the
model-only fields (`hook_patterns`, `formats`, `emotional_tone`,
`brand_application`) are returned empty with a "needs Gemini" note.
"""
from __future__ import annotations
import json
import threading

import config
from ingest import Video
from analyze import cluster_keywords

# Honest disclaimers - one per data situation, per language. Reused everywhere.
CAVEAT_NO_ENGAGEMENT = (
    "engagementデータ（再生数・いいね）が無いため、これは『使われている』表現で"
    "あって『伸びている』保証はありません。"
)
CAVEAT_WITH_ENGAGEMENT = (
    "再生数・いいね等は取得済み。ただしハッシュタグ検索で集めた一時点のスナップ"
    "ショット（時系列ではない）であり、再生数は累計・フォロワー数未補正。"
    "サンプル内での相対比較の目安として読み、『いま伸びている』とは断定しないこと。"
)
CAVEAT_NO_ENGAGEMENT_EN = (
    "No engagement data (views / likes), so this phrasing is only 'being used' - "
    "there is no evidence it is trending or high-performing."
)
CAVEAT_WITH_ENGAGEMENT_EN = (
    "View / like counts are present, but this is a one-shot hashtag-search "
    "snapshot (not a time series); counts are cumulative and not follower-"
    "normalised. Read it as a relative signal within this sample, never "
    "'trending right now'."
)

LANG_NAME = {"ja": "Japanese", "en": "English"}


def caveat_for(engagement_available: bool, language: str = "ja") -> str:
    if language == "en":
        return CAVEAT_WITH_ENGAGEMENT_EN if engagement_available else CAVEAT_NO_ENGAGEMENT_EN
    return CAVEAT_WITH_ENGAGEMENT if engagement_available else CAVEAT_NO_ENGAGEMENT


def caveat_no_engagement(language: str = "ja") -> str:
    return CAVEAT_NO_ENGAGEMENT_EN if language == "en" else CAVEAT_NO_ENGAGEMENT

_NEEDS_GEMINI = (
    "hook_patterns / formats / emotional_tone / brand_application は "
    "GEMINI_API_KEY を設定すると生成されます（フォールバックでは vocabulary と "
    "theme のみ機械集計）。"
)
_NEEDS_GEMINI_EN = (
    "hook_patterns / formats / emotional_tone / brand_application need "
    "GEMINI_API_KEY (the fallback only tallies vocabulary and theme)."
)


def _needs_gemini(language: str = "ja") -> str:
    return _NEEDS_GEMINI_EN if language == "en" else _NEEDS_GEMINI


# --- Evidence ---------------------------------------------------------------

def _cluster_evidence(
    videos: list[Video], max_items: int = 12, engagement_available: bool = False
) -> str:
    """Compact per-video sample sent to the model: every text signal we have.

    When engagement is available the sample is the TOP videos by view count and
    each line is annotated with views, so the model can weight its patterns
    toward what the higher-traffic clips in this sample actually say.
    """
    ordered = (
        sorted(videos, key=lambda v: v.engagement.get("views", 0), reverse=True)
        if engagement_available else videos
    )
    lines = []
    for v in ordered[:max_items]:
        parts = []
        if engagement_available and v.engagement.get("views"):
            parts.append(f"views: {v.engagement['views']:,}")
        if v.duration_sec:
            parts.append(f"dur: {v.duration_sec}s")
        if v.is_slideshow:
            parts.append("format: photo slideshow")
        if v.spoken_hook:
            parts.append(f"spoken_hook (verbatim, watched): {v.spoken_hook}")
        if v.on_screen_text:
            parts.append(f"on-screen: {v.on_screen_text}")
        if v.caption:
            parts.append(f"caption: {v.caption}")
        if v.transcript:
            parts.append(f"transcript: {v.transcript}")
        if v.hashtags:
            parts.append("hashtags: " + " ".join(f"#{h}" for h in v.hashtags))
        lines.append("- " + " || ".join(parts))
    return "\n".join(lines)


# --- Fallback: mechanical tallies ----------------------------------------------

def _frequent_terms(videos: list[Video], top_n: int = 15) -> list[str]:
    """Most frequent unigrams + bigrams across caption / on-screen / transcript.

    Uses sklearn's CountVectorizer (already a dependency) for tokenisation and
    English stop-word removal, so slang like "ghosted", "red flags",
    "per my last email", "cover letter" surfaces without any model call.
    """
    texts = [
        " ".join(t for t in (v.caption, v.on_screen_text, v.transcript) if t)
        for v in videos
    ]
    texts = [t for t in texts if t.strip()]
    if not texts:
        return []
    try:
        from sklearn.feature_extraction.text import CountVectorizer

        cv = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        matrix = cv.fit_transform(texts)
        freqs = matrix.sum(axis=0).A1
        ranked = sorted(
            zip(cv.get_feature_names_out(), freqs),
            key=lambda kv: kv[1],
            reverse=True,
        )
    except ValueError:
        return []

    repeated = [term for term, count in ranked if count >= 2]
    picked = repeated[:top_n] if repeated else [term for term, _ in ranked[:top_n]]
    return picked


def insight_fallback(
    videos: list[Video],
    engagement_available: bool = False,
    language: str = "ja",
    distinctive_hashtags: list[str] | None = None,
) -> dict:
    """No-API insight: fill vocabulary + theme from frequency counts only.

    `distinctive_hashtags`, when given (see
    analyze.distinctive_cluster_keywords), is used for the theme label instead
    of raw frequency - a hashtag-search scrape means the search seed tags
    dominate every cluster's raw top-N, so without this every cluster's theme
    looks nearly identical.
    """
    hashtags = cluster_keywords(videos, top_n=8)
    theme_hashtags = distinctive_hashtags if distinctive_hashtags else hashtags
    terms = _frequent_terms(videos)

    # De-dupe vocabulary: frequent phrases first, then any hashtags not already
    # covered by them.
    vocab = list(terms)
    lowered = {t.lower() for t in vocab}
    for h in hashtags:
        if h.lower() not in lowered:
            vocab.append(f"#{h}")

    if language == "en":
        theme_bits = " / ".join(theme_hashtags[:3]) if theme_hashtags else "(no hashtag info)"
        theme = f"{theme_bits} (hashtag-tally estimate; set a Gemini key for a real theme)"
    else:
        theme_bits = " / ".join(theme_hashtags[:3]) if theme_hashtags else "（ハッシュタグ情報なし）"
        theme = f"{theme_bits}（ハッシュタグ集計ベースの推定。Geminiキーで具体化されます）"

    return {
        "theme": theme,
        "hook_patterns": [],
        "formats": [],
        "vocabulary": vocab[:20],
        "emotional_tone": [],
        "brand_application": [],
        "caveat": caveat_for(engagement_available, language),
        "needs_gemini_note": _needs_gemini(language),
        "source": "fallback",
        "hook_source": "none",
    }


# --- Normalisation of a model-produced insight --------------------------------

_EVIDENCE_PREFIXES = (
    "views:", "dur:", "format:", "spoken_hook (verbatim, watched):",
    "on-screen:", "caption:", "transcript:", "hashtags:",
)


def _strip_evidence_prefixes(text: str) -> str:
    """Remove the 'on-screen:' / 'caption:' ... scaffolding that the model
    sometimes copies verbatim out of the evidence block into a quote."""
    text = str(text).strip()
    # Split on the '||' evidence separator and keep the most quotable segment:
    # on-screen text, else caption, else the first segment that isn't just a
    # metric like "views: 1,234" / "dur: 60s".
    _metric = ("views:", "dur:", "format:")
    if "||" in text:
        segments = [s.strip() for s in text.split("||") if s.strip()]
        chosen = (
            next((s for s in segments if s.lower().startswith("spoken_hook")), None)
            or next((s for s in segments if s.lower().startswith("on-screen:")), None)
            or next((s for s in segments if s.lower().startswith("caption:")), None)
            or next((s for s in segments if not s.lower().startswith(_metric)), None)
            or (segments[0] if segments else text)
        )
        text = chosen
    low = text.lower()
    for p in _EVIDENCE_PREFIXES:
        if low.startswith(p):
            text = text[len(p):].strip()
            break
    return text


def _norm_str_list(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    cleaned = [_strip_evidence_prefixes(x) for x in value]
    return [x for x in cleaned if x]


def _normalise_insight(
    item: dict,
    engagement_available: bool = False,
    language: str = "ja",
    has_video_evidence: bool = False,
) -> dict:
    """Coerce one raw model object into the canonical insight shape."""
    hook_patterns = []
    for hp in item.get("hook_patterns", [])[:6]:
        if not isinstance(hp, dict):
            continue
        pattern = _strip_evidence_prefixes(hp.get("pattern", ""))
        example = _strip_evidence_prefixes(hp.get("example", ""))
        if pattern or example:
            hook_patterns.append({"pattern": pattern, "example": example})

    applications = []
    for ap in item.get("brand_application", [])[:4]:
        if not isinstance(ap, dict):
            continue
        hook = _strip_evidence_prefixes(ap.get("suggested_hook", ""))
        angle = str(ap.get("angle", "")).strip()
        tool = str(ap.get("tool", "")).strip()
        if hook or angle:
            applications.append(
                {"suggested_hook": hook, "angle": angle, "tool": tool}
            )

    no_theme = "(theme not returned)" if language == "en" else "（テーマ未取得）"
    return {
        "theme": str(item.get("theme", "")).strip() or no_theme,
        "hook_patterns": hook_patterns,
        "formats": _norm_str_list(item.get("formats")),
        "vocabulary": _norm_str_list(item.get("vocabulary")),
        "emotional_tone": _norm_str_list(item.get("emotional_tone")),
        "brand_application": applications,
        "caveat": str(item.get("caveat", "")).strip()
        or caveat_for(engagement_available, language),
        "source": "gemini",
        "hook_source": "video" if has_video_evidence else "caption",
    }


# --- Primary path: one Gemini call for every in-scope cluster -----------------

def _response_schema():
    """OpenAPI-subset schema for the batch response: a JSON ARRAY, one object
    per cluster. A fixed array beats a dynamic-key {"<id>": ...} map, which
    Gemini's structured-output can't validate. Built as a protos.Schema so it
    works without pulling in Pydantic.
    """
    import google.generativeai as genai
    S, T = genai.protos.Schema, genai.protos.Type

    strings = S(type=T.ARRAY, items=S(type=T.STRING))
    hook = S(type=T.OBJECT, properties={
        "pattern": S(type=T.STRING),
        "example": S(type=T.STRING),
    }, required=["pattern", "example"])
    application = S(type=T.OBJECT, properties={
        "tool": S(type=T.STRING),
        "suggested_hook": S(type=T.STRING),
        "angle": S(type=T.STRING),
    }, required=["tool", "suggested_hook", "angle"])
    cluster = S(type=T.OBJECT, properties={
        "cluster_id": S(type=T.STRING),
        "theme": S(type=T.STRING),
        "hook_patterns": S(type=T.ARRAY, items=hook),
        "formats": strings,
        "vocabulary": strings,
        "emotional_tone": strings,
        "brand_application": S(type=T.ARRAY, items=application),
        "caveat": S(type=T.STRING),
    }, required=[
        "cluster_id", "theme", "hook_patterns", "formats", "vocabulary",
        "emotional_tone", "brand_application", "caveat",
    ])
    return S(type=T.ARRAY, items=cluster)


def extract_insights_batch(
    clusters_in_scope: dict[int, list[Video]],
    brand: dict,
    engagement_available: bool = False,
    language: str = "ja",
) -> dict[int, dict]:
    """ONE Gemini call producing a language insight for every given cluster.

    `clusters_in_scope` is already filtered by the pipeline's relevance
    gatekeeper, so we only spend the call on job-hunt / career clusters.

    The call pins response_mime_type=application/json + a response_schema, so
    the model returns a valid, fully-keyed JSON array rather than free-form
    text - no code-fence stripping, no missing keys, no mid-JSON truncation.
    A hard wall-clock timeout (config.GEMINI_TIMEOUT, enforced with a worker
    thread because the SDK's own request timeout is unreliable) means a hung
    or endlessly-retrying call can't stall the run.

    Returns {cluster_id: insight dict}. Raises on any failure (missing
    package, network, timeout, bad JSON) so the caller can fall back locally
    without making another call.
    """
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)

    lang_name = LANG_NAME.get(language, "Japanese")
    tools = "\n".join(f"- {t}" for t in brand.get("tools", []))
    blocks = []
    for cid, videos in clusters_in_scope.items():
        kw = ", ".join(cluster_keywords(videos)) or "(none)"
        blocks.append(
            f"### Cluster {cid} ({len(videos)} videos)\n"
            f"Top hashtags: {kw}\n"
            f"Videos:\n"
            f"{_cluster_evidence(videos, engagement_available=engagement_available)}"
        )
    clusters_text = "\n\n".join(blocks)

    if engagement_available:
        honesty_rule = (
            "IMPORTANT HONESTY RULE: each video line carries a `views` count. "
            "You MAY lean your hook_patterns/formats toward what the "
            "higher-view videos in the sample say, and may note which phrasings "
            "recur among them. But this is a hashtag-search snapshot, not a "
            "time series, and view counts are cumulative and not "
            "follower-normalised - so NEVER say a phrasing is going viral, "
            "trending right now, or guaranteed to perform. Each cluster's "
            f"`caveat` must state this limitation in {lang_name}."
        )
    else:
        honesty_rule = (
            "IMPORTANT HONESTY RULE: the source data has NO engagement metrics "
            "(no views, no likes). You may only say a phrasing/format is BEING "
            "USED in this niche. Never claim anything is viral, blowing up, "
            "trending, or high-performing. Each cluster's `caveat` must state "
            f"this limitation in {lang_name}."
        )

    prompt = (
        "You are a short-form social media strategist for the brand below. "
        "Marketers will use your output to write Instagram Reels / TikTok "
        "scripts, so be concrete: quote real phrasings, name real formats.\n\n"
        f"{honesty_rule}\n\n"
        f"Brand: {brand.get('name', '')}\n"
        f"Brand description: {brand.get('description', '')}\n"
        f"Target audience: {brand.get('target_audience', '')}\n"
        f"Brand tone: {brand.get('tone', '')}\n"
        f"Brand tools (use EXACT names when referencing one):\n{tools}\n\n"
        f"Write ALL prose fields (theme, emotional_tone, angle, caveat) in "
        f"{lang_name}. Keep quoted phrasings (hook_patterns.example, vocabulary, "
        f"suggested_hook) in their ORIGINAL language as spoken in the videos.\n\n"
        "Below are clusters of job-hunt / career short-form videos that an "
        "algorithm grouped as similar. For EACH cluster, extract:\n"
        f"- theme: the job-hunt topic this cluster represents ({lang_name}, "
        "short)\n"
        "- hook_patterns: 3-5 hook templates ACTUALLY used in this cluster. "
        'Each = {"pattern": template with slots e.g. "POV: {situation}", '
        '"example": a real line taken from this cluster}. Some videos below '
        "carry a `spoken_hook (verbatim, watched)` line - the actual first "
        "line spoken/shown in that video, from someone who watched it. "
        "Whenever spoken_hook lines are present in a cluster, build "
        "hook_patterns.example ONLY from those verbatim lines, never from "
        "caption text (a caption is marketing copy written to sit next to "
        "the video, not necessarily what's said or shown IN it - do not "
        "treat it as a hook). Only fall back to inferring from caption "
        "phrasing for clusters with no spoken_hook lines at all.\n"
        "- formats: video formats used (e.g. \"talking head + text overlay\", "
        "\"POV skit\", \"GRWM\"), most frequent first\n"
        "- vocabulary: recurring slang / phrases / keywords buyers of this "
        'content would recognise (e.g. "ghosted", "red flags", "delulu", '
        '"per my last email"). Keep the original language of the phrase.\n'
        f"- emotional_tone: the emotional tones that land here ({lang_name})\n"
        "- brand_application: EXACTLY 2 post ideas that repurpose this "
        "cluster's language/format to promote one of the brand's tools. Each = "
        '{"tool": exact tool name, "suggested_hook": the brand version '
        'of a hook (one line, ready to film), "angle": the marketing angle in '
        f"{lang_name}}}\n"
        "- cluster_id: the cluster number this object is for, as a string\n"
        f"- caveat: the {lang_name} engagement disclaimer described above\n\n"
        "In every quote you copy (example, suggested_hook, vocabulary), strip "
        "the scaffolding labels 'views:', 'dur:', 'on-screen:', 'caption:', "
        "'transcript:', 'hashtags:' and the '||' separators - quote only the "
        "human text.\n\n"
        f"Clusters:\n\n{clusters_text}\n\n"
        "Return a JSON array with one object per cluster above (cluster_id "
        "matching the cluster numbers)."
    )

    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=_response_schema(),
    )

    # google-generativeai's request_options timeout is unreliable (the SDK's
    # own retry loop backs off well past it on a rate-limit wall). Bound the
    # whole call from outside with a DAEMON worker thread: on timeout we raise
    # and fall back, and because the thread is a daemon a still-hung SDK call
    # (e.g. an SSL handshake with no route) can't keep the process alive at
    # exit. ThreadPoolExecutor's workers are non-daemon and would do exactly
    # that, so we spin the thread by hand.
    box: dict[str, object] = {}

    def _do_call():
        try:
            box["resp"] = model.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": config.GEMINI_TIMEOUT},
            )
        except BaseException as exc:  # noqa: BLE001 - relayed to the caller below
            box["err"] = exc

    worker = threading.Thread(target=_do_call, name="gemini-insight", daemon=True)
    worker.start()
    worker.join(timeout=config.GEMINI_TIMEOUT)
    if worker.is_alive():
        raise TimeoutError(
            f"Gemini call exceeded {config.GEMINI_TIMEOUT}s wall-clock limit"
        )
    if "err" in box:
        raise box["err"]  # type: ignore[misc]
    resp = box["resp"]

    text = (resp.text or "").strip()
    if text.startswith("```"):  # belt-and-suspenders; schema mode shouldn't fence
        text = text.strip("`")
        text = text[text.find("["):]
    data = json.loads(text)

    by_id: dict[str, dict] = {}
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and str(item.get("cluster_id", "")).strip():
            by_id[str(item["cluster_id"]).strip()] = item

    return {
        cid: _normalise_insight(
            by_id.get(str(cid), {}),
            engagement_available,
            language,
            has_video_evidence=any(v.spoken_hook for v in clusters_in_scope[cid]),
        )
        for cid in clusters_in_scope
    }
