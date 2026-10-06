"""
Explanation layer: give each cluster a human-readable name and a short
explanation of the pattern it represents. This is the "intelligence" the
prototype is meant to prove.

Primary path: Gemini (free tier). Given a cluster's captions/on-screen text,
it returns a short label like "Recruiter ghosting humor" plus a one-line
rationale.

Fallback path: if there's no API key or the package/network is unavailable,
we degrade gracefully to a keyword-based label so the pipeline still produces
an end-to-end result. The mentor's rule was: prove the pipeline runs first.
"""
from __future__ import annotations
import json

import config
from ingest import Video
from analyze import cluster_keywords


def explain_cluster(videos: list[Video]) -> dict:
    """Return {'label': str, 'why': str, 'source': 'gemini'|'fallback'}."""
    if config.GEMINI_API_KEY:
        try:
            return _explain_with_gemini(videos)
        except Exception as e:
            # Never let a bad API call kill the run; fall back and note why.
            fb = explain_fallback(videos)
            fb["why"] += f"  (Gemini unavailable: {type(e).__name__})"
            return fb
    return explain_fallback(videos)


def _cluster_evidence(videos: list[Video], max_items: int = 12) -> str:
    """Compact text sample of a cluster to send to the LLM."""
    lines = []
    for v in videos[:max_items]:
        bits = [b for b in (v.on_screen_text, v.caption) if b]
        lines.append("- " + " | ".join(bits))
    return "\n".join(lines)


def _explain_with_gemini(videos: list[Video]) -> dict:
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)

    evidence = _cluster_evidence(videos)
    keywords = ", ".join(cluster_keywords(videos)) or "(none)"

    prompt = (
        "You are a social media trend analyst. Below is a group of short-form "
        "videos that an algorithm clustered together because they seem similar.\n"
        "Give this group a short, punchy NAME (3-6 words, like a trend label) "
        "and a ONE-SENTENCE explanation of the shared pattern (topic, format, "
        "emotion, or behavior).\n\n"
        f"Common hashtags: {keywords}\n\n"
        f"Videos (on-screen text | caption):\n{evidence}\n\n"
        "Respond as strict JSON only, no markdown, in this exact shape:\n"
        '{"label": "...", "why": "..."}'
    )

    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()
    # Strip accidental code fences if the model adds them.
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    data = json.loads(text)
    return {
        "label": str(data.get("label", "")).strip() or "Unnamed pattern",
        "why": str(data.get("why", "")).strip(),
        "source": "gemini",
    }


def explain_fallback(videos: list[Video]) -> dict:
    """Keyword-based label so the pipeline yields output with no API."""
    kws = cluster_keywords(videos, top_n=3)
    label = " / ".join(kws) if kws else "Unlabeled pattern"
    why = (
        f"{len(videos)} videos sharing hashtags: "
        + (", ".join(kws) if kws else "n/a")
        + ". (Keyword fallback - set GEMINI_API_KEY for a real explanation.)"
    )
    return {"label": label, "why": why, "source": "fallback"}
