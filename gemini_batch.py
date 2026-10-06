"""
Single-call Gemini orchestration.

explain.py, brand_match.py and generate.py each know how to do their job
for ONE cluster via its own Gemini call - useful in isolation, but calling
all three per cluster means N clusters cost up to 3N API calls, which blows
through the free-tier quota fast (see: 6 clusters -> 12+ calls hit
ResourceExhausted).

This module trades a little prompt complexity for that: it asks Gemini to
explain, brand-score, and (where relevant) draft content ideas for EVERY
cluster in ONE request, so a pipeline run costs at most one Gemini call
total. If that call fails (or no key is set), pipeline.py falls back to the
local, no-API fallback logic in explain.py / brand_match.py / generate.py
for every cluster - so a failure here never triggers extra API calls.
"""
from __future__ import annotations
import json

import config
from ingest import Video
from analyze import cluster_keywords


def _cluster_evidence(videos: list[Video], max_items: int = 8) -> str:
    lines = []
    for v in videos[:max_items]:
        bits = [b for b in (v.on_screen_text, v.caption) if b]
        lines.append("- " + " | ".join(bits))
    return "\n".join(lines)


def run_batch(
    clusters: dict[int, list[Video]],
    brand: dict,
    relevance_threshold: int,
) -> dict[int, dict]:
    """One Gemini call covering every cluster.

    Returns {cluster_id: {label, why, relevance_score, reason,
    recommended_platform, matched_tools, ideas}}.

    Raises on any failure (missing package, network error, bad JSON) so the
    caller can fall back to the local per-cluster logic without making
    another API call.
    """
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)

    tools = "\n".join(f"- {t}" for t in brand.get("tools", []))
    cluster_blocks = []
    for cid, videos in clusters.items():
        keywords = ", ".join(cluster_keywords(videos)) or "(none)"
        cluster_blocks.append(
            f"Cluster {cid} ({len(videos)} videos)\n"
            f"Hashtags: {keywords}\n"
            f"Videos (on-screen text | caption):\n{_cluster_evidence(videos)}"
        )
    clusters_text = "\n\n".join(cluster_blocks)

    prompt = (
        "You are a social media trend analyst AND brand strategist. Below are "
        "several clusters of short-form videos, each grouped by an algorithm "
        "because the videos in it seem similar to each other.\n\n"
        "For EACH cluster, do all of the following:\n"
        "1. Give it a short, punchy NAME (3-6 words, like a trend label) and a "
        "ONE-SENTENCE explanation of the shared pattern (topic, format, "
        "emotion, or behavior).\n"
        "2. Score 0-100 how relevant this trend is for the brand below to "
        "create content about (100 = perfect fit, 0 = unrelated), with a "
        "one-sentence reason.\n"
        "3. Recommend ONE platform for this content: \"TikTok\" or "
        "\"Instagram Reels\".\n"
        "4. List at most 2 brand tools (exact names from the list below) "
        "relevant to this trend - empty list if none fit.\n"
        f"5. ONLY IF relevance_score >= {relevance_threshold}: write 2-3 short-form "
        "content ideas (~15 seconds each) in the brand's voice riding this "
        "trend, each with a hook (first ~3 seconds), a short script, and a "
        "caption with hashtags. Keep it authentic, not salesy. If "
        "relevance_score is below the threshold, return an empty ideas list "
        "for that cluster - do not write ideas for it.\n\n"
        f"Brand: {brand.get('name', '')}\n"
        f"Brand description: {brand.get('description', '')}\n"
        f"Target audience: {brand.get('target_audience', '')}\n"
        f"Brand tone: {brand.get('tone', '')}\n"
        f"Brand tools:\n{tools}\n\n"
        f"Clusters:\n\n{clusters_text}\n\n"
        "Respond as strict JSON only, no markdown, in this exact shape "
        "(cluster_id keys as strings, matching the cluster numbers above):\n"
        '{"clusters": {"<cluster_id>": {"label": "...", "why": "...", '
        '"relevance_score": 0, "reason": "...", "recommended_platform": "...", '
        '"matched_tools": ["..."], '
        '"ideas": [{"hook": "...", "script": "...", "caption": "..."}]}}}'
    )

    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    data = json.loads(text)

    raw_clusters = data.get("clusters", {})
    result: dict[int, dict] = {}
    for cid in clusters:
        item = raw_clusters.get(str(cid), {})

        score = max(0, min(100, int(item.get("relevance_score", 0))))
        platform = str(item.get("recommended_platform", "TikTok")).strip()
        if platform not in ("TikTok", "Instagram Reels"):
            platform = "TikTok"
        ideas = []
        for idea in item.get("ideas", [])[:3]:
            ideas.append({
                "hook": str(idea.get("hook", "")).strip(),
                "script": str(idea.get("script", "")).strip(),
                "caption": str(idea.get("caption", "")).strip(),
            })

        result[cid] = {
            "label": str(item.get("label", "")).strip() or "Unnamed pattern",
            "why": str(item.get("why", "")).strip(),
            "relevance_score": score,
            "reason": str(item.get("reason", "")).strip(),
            "recommended_platform": platform,
            "matched_tools": [str(t) for t in item.get("matched_tools", [])][:2],
            "ideas": ideas,
        }
    return result
