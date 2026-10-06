"""
Content generation layer: for clusters that scored well on brand fit
(brand_match.py), draft a handful of short-form content ideas the brand
could post to ride the trend.

Follows the same Gemini primary / graceful-fallback pattern as explain.py
and brand_match.py: if there's no API key, we skip generation rather than
fail the pipeline, and say so explicitly in the output.
"""
from __future__ import annotations
import json

import config
from ingest import Video


def skipped(note: str) -> dict:
    """Shape used whenever no content ideas were generated (no key, below
    threshold, or a batched Gemini call failed elsewhere in the pipeline)."""
    return {"ideas": [], "source": "skipped", "note": note}


def generate_ideas(
    videos: list[Video],
    brand: dict,
    cluster_label: str,
    cluster_why: str,
    matched_tools: list[str],
    n_ideas: int = 3,
) -> dict:
    """Return {'ideas': list[{'hook','script','caption'}], 'source': 'gemini'|'skipped'}."""
    if not config.GEMINI_API_KEY:
        return skipped("GEMINI_API_KEY not set - content generation skipped.")
    try:
        return _generate_with_gemini(videos, brand, cluster_label, cluster_why, matched_tools, n_ideas)
    except Exception as e:
        return skipped(f"Gemini unavailable: {type(e).__name__}")


def _cluster_evidence(videos: list[Video], max_items: int = 8) -> str:
    lines = []
    for v in videos[:max_items]:
        bits = [b for b in (v.on_screen_text, v.caption) if b]
        lines.append("- " + " | ".join(bits))
    return "\n".join(lines)


def _generate_with_gemini(
    videos: list[Video],
    brand: dict,
    cluster_label: str,
    cluster_why: str,
    matched_tools: list[str],
    n_ideas: int,
) -> dict:
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)

    evidence = _cluster_evidence(videos)
    tools_str = ", ".join(matched_tools) if matched_tools else "(none specific - keep it general)"

    prompt = (
        "You are a short-form video content writer for a brand, riding an "
        "existing trend without sounding like an ad.\n\n"
        f"Brand: {brand.get('name', '')}\n"
        f"Brand description: {brand.get('description', '')}\n"
        f"Brand tone: {brand.get('tone', '')}\n"
        f"Target audience: {brand.get('target_audience', '')}\n"
        f"Relevant brand tools to weave in naturally (don't force all of them): {tools_str}\n\n"
        f"Trend name: {cluster_label}\n"
        f"Trend pattern: {cluster_why}\n"
        f"Example videos in this trend (on-screen text | caption):\n{evidence}\n\n"
        f"Write {n_ideas} distinct short-form video ideas (~15 seconds each) that "
        "ride this trend in the brand's voice. Each idea needs:\n"
        "- hook: the first ~3 seconds of spoken/on-screen line, must grab attention\n"
        "- script: a short ~15 second script/beat outline\n"
        "- caption: a post caption including relevant hashtags\n\n"
        "Keep it authentic and useful, not salesy. Respond as strict JSON only, "
        "no markdown, in this exact shape:\n"
        '{"ideas": [{"hook": "...", "script": "...", "caption": "..."}]}'
    )

    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    data = json.loads(text)

    ideas = []
    for item in data.get("ideas", [])[:n_ideas]:
        ideas.append({
            "hook": str(item.get("hook", "")).strip(),
            "script": str(item.get("script", "")).strip(),
            "caption": str(item.get("caption", "")).strip(),
        })

    return {"ideas": ideas, "source": "gemini"}
