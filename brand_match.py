"""
Brand-fit layer: score how relevant each discovered cluster (trend) is to a
given brand, following the same "Gemini primary / keyword fallback" pattern
as explain.py.

Brand identity itself is deliberately kept out of this module (it lives in
a JSON file, see load_brand()) so a new brand is a data swap, not a code
change - the same "swappable" spirit as ingest.py's Video source.
"""
from __future__ import annotations
import json
import re

import config
from ingest import Video
from analyze import cluster_keywords


def _cluster_blob(videos: list[Video]) -> str:
    """All lowercased text signals of a cluster, joined for keyword scanning."""
    parts: list[str] = []
    for v in videos:
        parts.append(v.caption.lower())
        parts.append(v.on_screen_text.lower())
        parts.append(v.transcript.lower())
        parts.extend(h.lower() for h in v.hashtags)
    return "  ".join(p for p in parts if p)


def _keyword_hits(blob: str, brand_keywords: list[str]) -> set[str]:
    """Brand keywords present in `blob` as WHOLE words.

    Whole-word matching matters for scraped hashtag data: real job-hunt clips
    tag `#ghosted`, but Roblox clips tag `#ghostedthecall` and truck-sim clips
    tag `#ats` (American Truck Simulator). Substring matching would wave those
    false positives through; `(?<!\\w)kw(?!\\w)` does not.
    """
    hits: set[str] = set()
    for kw in brand_keywords:
        if re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", blob):
            hits.add(kw)
    return hits


def load_brand(path: str) -> dict:
    """Load a brand profile from a JSON file (name/description/tone/tools/keywords)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_cluster(videos: list[Video], cluster_label: str, cluster_why: str, brand: dict) -> dict:
    """Return {'relevance_score': int, 'reason': str, 'recommended_platform': str,
    'matched_tools': list[str], 'source': 'gemini'|'fallback'}."""
    if config.GEMINI_API_KEY:
        try:
            return _match_with_gemini(videos, cluster_label, cluster_why, brand)
        except Exception as e:
            fb = match_fallback(videos, brand)
            fb["reason"] += f"  (Gemini unavailable: {type(e).__name__})"
            return fb
    return match_fallback(videos, brand)


def _cluster_evidence(videos: list[Video], max_items: int = 12) -> str:
    lines = []
    for v in videos[:max_items]:
        bits = [b for b in (v.on_screen_text, v.caption) if b]
        lines.append("- " + " | ".join(bits))
    return "\n".join(lines)


def _match_with_gemini(videos: list[Video], cluster_label: str, cluster_why: str, brand: dict) -> dict:
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)

    evidence = _cluster_evidence(videos)
    tools = "\n".join(f"- {t}" for t in brand.get("tools", []))

    prompt = (
        "You are a brand social-media strategist deciding whether a discovered "
        "content trend is worth this brand riding on.\n\n"
        f"Brand: {brand.get('name', '')}\n"
        f"Brand description: {brand.get('description', '')}\n"
        f"Target audience: {brand.get('target_audience', '')}\n"
        f"Brand tools:\n{tools}\n\n"
        f"Trend name: {cluster_label}\n"
        f"Trend pattern: {cluster_why}\n"
        f"Trend videos (on-screen text | caption):\n{evidence}\n\n"
        "Score how relevant this trend is for the brand to create content about "
        "(0-100, where 100 = perfect fit for the brand's audience and product, "
        "0 = totally unrelated). Pick ONE recommended platform for this specific "
        "content: \"TikTok\" or \"Instagram Reels\". List at most 2 brand tools "
        "(from the list above, use their exact names) that are actually relevant "
        "to this trend - empty list if none fit.\n\n"
        "Respond as strict JSON only, no markdown, in this exact shape:\n"
        '{"relevance_score": 0, "reason": "...", "recommended_platform": "...", '
        '"matched_tools": ["..."]}'
    )

    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    data = json.loads(text)

    score = int(data.get("relevance_score", 0))
    score = max(0, min(100, score))
    platform = str(data.get("recommended_platform", "TikTok")).strip()
    if platform not in ("TikTok", "Instagram Reels"):
        platform = "TikTok"
    matched = [str(t) for t in data.get("matched_tools", [])][:2]

    return {
        "relevance_score": score,
        "reason": str(data.get("reason", "")).strip(),
        "recommended_platform": platform,
        "matched_tools": matched,
        "source": "gemini",
    }


def gatekeeper_score(videos: list[Video], brand: dict) -> int:
    """Cheap, no-API relevance score (0-100) used ONLY as an internal gate:
    which clusters are on-topic enough (job hunt / career) to be worth
    generating a language insight for.

    This is deliberately NOT the headline output anymore - it just filters.
    Score = share of the cluster's videos that mention ANY brand keyword (as a
    whole word) in their hashtags, caption or on-screen text. A job-hunt cluster
    lands high; a "girl dinner" / "soft life" cluster lands near zero.

    Pair it with distinct_brand_keywords(): a cluster clears the gate only when
    it also hits >= config.MIN_DISTINCT_KEYWORDS different brand keywords, so a
    lone accidental collision (`#ats` = American Truck Simulator) can't let an
    off-topic cluster through.
    """
    brand_keywords = [k.lower() for k in brand.get("keywords", [])]
    if not brand_keywords or not videos:
        return 0

    hits = sum(
        1 for v in videos if _keyword_hits(_cluster_blob([v]), brand_keywords)
    )
    return round(hits / len(videos) * 100)


def video_is_on_topic(video: Video, brand: dict) -> bool:
    """True if THIS one video individually hits >= 1 brand keyword (whole
    word) in its own hashtags/caption/on-screen text.

    gatekeeper_score()/distinct_brand_keywords() only judge a CLUSTER in
    aggregate, so a single off-topic video (hashtag spam - e.g. an unrelated
    clip tagged #jobsearch for reach, or one that drifted into the cluster by
    embedding similarity alone) can ride along inside an otherwise on-topic,
    passing cluster and still surface as a "representative example". Filter
    a cluster's own video list through this before picking examples/evidence
    to drop those individual outliers without re-litigating the whole
    cluster's gate.
    """
    brand_keywords = [k.lower() for k in brand.get("keywords", [])]
    if not brand_keywords:
        return False
    return bool(_keyword_hits(_cluster_blob([video]), brand_keywords))


def distinct_brand_keywords(videos: list[Video], brand: dict) -> list[str]:
    """Which distinct brand keywords the whole cluster hits (whole-word)."""
    brand_keywords = [k.lower() for k in brand.get("keywords", [])]
    if not brand_keywords or not videos:
        return []
    return sorted(_keyword_hits(_cluster_blob(videos), brand_keywords))


def match_fallback(videos: list[Video], brand: dict) -> dict:
    """Keyword-overlap heuristic so the pipeline still yields a score with no API.

    Score = (# of brand keywords present in the cluster's hashtags) /
            (# of brand keywords) * 100, capped at 100.
    """
    brand_keywords = {k.lower() for k in brand.get("keywords", [])}
    cluster_hashtags = {h.lower() for h in cluster_keywords(videos, top_n=1000)}

    overlap = brand_keywords & cluster_hashtags
    score = 0
    if brand_keywords:
        score = round(len(overlap) / len(brand_keywords) * 100)
    score = max(0, min(100, score))

    if overlap:
        reason = (
            f"{len(overlap)}件のブランドキーワードとハッシュタグが一致: "
            + ", ".join(sorted(overlap))
            + "。(Keyword fallback - set GEMINI_API_KEY for a real assessment.)"
        )
    else:
        reason = (
            "ブランドキーワードとの一致なし。"
            "(Keyword fallback - set GEMINI_API_KEY for a real assessment.)"
        )

    return {
        "relevance_score": score,
        "reason": reason,
        "recommended_platform": "TikTok",
        "matched_tools": [],
        "source": "fallback",
    }
