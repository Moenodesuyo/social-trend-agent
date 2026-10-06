"""
Self-audit layer: score the brand's OWN posted videos against the language
patterns the pipeline extracted from the job-hunt / career niche.

Flow (all in the Streamlit "自己診断" tab):

    upload own .mp4(s)
      -> analyse_video_file(): ONE Gemini call per clip, video passed straight
         to the model (File API). Pulls spoken hook, transcript, on-screen text,
         format, visual tags.  -> a canonical ingest.Video
      -> reference_patterns(): merge the in-scope cluster insights from a normal
         pipeline run into one "what the niche is doing" pattern sheet
      -> audit_against_reference(): ONE batched Gemini call comparing every own
         clip to that sheet -> per-clip {assessment, what_works, gaps,
         hook_rewrites, format_suggestion, vocabulary_to_adopt, priority}

Honesty guardrail (same as insight.py): the own clips carry no engagement, and
the reference is a hashtag-search snapshot, so every audit item carries a
`caveat` and nothing claims a rewrite will "go viral" - only that it moves the
post toward phrasing/formats the niche actually uses.

No key / call failed -> mechanical fallback: a vocabulary set-difference
(reference terms the clip does / doesn't already use) plus the reference hook
templates to consider. The model-only fields come back empty with a note.
"""
from __future__ import annotations
import json
import os
import threading
import time
from collections import Counter
from dataclasses import replace
from urllib.parse import urlparse

import config
from ingest import Video
from insight import caveat_for, LANG_NAME
import sound_trends

_NEEDS_GEMINI = {
    "ja": (
        "assessment / gaps / hook_rewrites / format_suggestion は "
        "GEMINI_API_KEY を設定すると生成されます（フォールバックでは参照語彙との"
        "差分のみ）。"
    ),
    "en": (
        "assessment / gaps / hook_rewrites / format_suggestion need "
        "GEMINI_API_KEY (the fallback is only a vocabulary set-difference)."
    ),
}

# UI-language phrasing for the mechanical fallback (no Gemini call).
_FB = {
    "ja": {
        "matched": "参照語彙と一致: {v}",
        "missing": "参照で頻出だが未使用の語彙: {v}",
        "ref_formats": "参照フォーマット: {v}",
    },
    "en": {
        "matched": "Already uses niche vocabulary: {v}",
        "missing": "Niche vocabulary not used yet: {v}",
        "ref_formats": "Niche formats: {v}",
    },
}


# --- Reference pattern sheet -------------------------------------------------

def _dedup(seq, cap: int):
    out, seen = [], set()
    for x in seq:
        key = x.lower() if isinstance(x, str) else json.dumps(x, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
        if len(out) >= cap:
            break
    return out


def reference_patterns(report: list[dict]) -> dict:
    """Merge every in-scope cluster's insight into one pattern sheet.

    `report` is exactly what pipeline.run() returns. Only clusters that cleared
    the relevance gate (`is_target`) and actually got an insight contribute.
    """
    themes, hooks, formats, vocab, tones = [], [], [], [], []
    for c in report:
        if not c.get("is_target"):
            continue
        ins = c.get("insight") or {}
        if ins.get("theme"):
            themes.append(ins["theme"])
        for hp in ins.get("hook_patterns", []):
            if hp.get("pattern"):
                hooks.append({"pattern": hp["pattern"], "example": hp.get("example", "")})
        formats.extend(ins.get("formats", []))
        vocab.extend(ins.get("vocabulary", []))
        tones.extend(ins.get("emotional_tone", []))
    return {
        "themes": _dedup(themes, 12),
        "hook_patterns": _dedup(hooks, 15),
        "formats": _dedup(formats, 12),
        "vocabulary": _dedup(vocab, 40),
        "emotional_tone": _dedup(tones, 10),
    }


# --- Bounded Gemini call (mirrors insight.py's daemon-thread guard) ----------

def _call_with_timeout(fn, timeout: int):
    box: dict[str, object] = {}

    def _run():
        try:
            box["out"] = fn()
        except BaseException as exc:  # noqa: BLE001 - relayed to caller
            box["err"] = exc

    worker = threading.Thread(target=_run, name="gemini-video", daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        raise TimeoutError(f"Gemini call exceeded {timeout}s wall-clock limit")
    if "err" in box:
        raise box["err"]  # type: ignore[misc]
    return box["out"]


# --- Real engagement (Apify) merge ------------------------------------------

def _has_real_engagement(v: Video) -> bool:
    """True when `v.engagement` came from a live Apify scrape, not just a
    number the user typed into the audit form. Uses the explicit
    `engagement_is_real` flag rather than guessing from which keys are
    present - a reel with likes hidden by the creator legitimately has only
    `views` in `engagement`, same shape as a manual view-count entry, so the
    key set alone can't tell the two apart."""
    return v.engagement_is_real and bool(v.engagement)


def _url_key(url: str) -> str:
    """Normalise a post URL to a stable matching key (last non-empty path
    segment, no query string / trailing slash) so a pasted URL matches the
    Apify row regardless of query params or protocol."""
    if not url:
        return ""
    segments = [s for s in urlparse(url.strip()).path.split("/") if s]
    return segments[-1].lower() if segments else url.strip().lower()


def merge_apify_engagement(
    own_videos: list[Video], apify_videos: list[Video]
) -> tuple[list[Video], int]:
    """Fold REAL engagement (+ sound + timestamp) pulled live from Apify into
    the Gemini-read own_videos, matched by post URL. Pixel-derived fields
    (transcript, spoken_hook, detected_format, on_screen_text, visual_tags)
    always stay from the Gemini video read - Apify has no video access - only
    engagement / url / timestamp / sound are overwritten on a match.

    Returns (merged_videos, n_matched), same order/length as `own_videos`.
    """
    by_key = {_url_key(v.url): v for v in apify_videos if v.url}
    merged: list[Video] = []
    n_matched = 0
    for v in own_videos:
        match = by_key.get(_url_key(v.url)) if v.url else None
        if match is None:
            merged.append(v)
            continue
        n_matched += 1
        merged.append(replace(
            v,
            url=match.url or v.url,
            engagement=dict(match.engagement),
            engagement_is_real=True,
            sound_id=match.sound_id or v.sound_id,
            sound_name=match.sound_name or v.sound_name,
            sound_is_original=match.sound_is_original,
            timestamp=match.timestamp or v.timestamp,
        ))
    return merged, n_matched


# --- Step 1: one video -> one Video ----------------------------------------

def _analysis_schema():
    import google.generativeai as genai
    S, T = genai.protos.Schema, genai.protos.Type
    return S(type=T.OBJECT, properties={
        "spoken_hook": S(type=T.STRING),
        "transcript": S(type=T.STRING),
        "on_screen_text": S(type=T.STRING),
        "format": S(type=T.STRING),
        "visual_tags": S(type=T.ARRAY, items=S(type=T.STRING)),
        "duration_sec": S(type=T.INTEGER),
        "is_slideshow": S(type=T.BOOLEAN),
    }, required=[
        "spoken_hook", "transcript", "on_screen_text", "format",
        "visual_tags", "duration_sec", "is_slideshow",
    ])


_ANALYSIS_PROMPT = (
    "You are analysing ONE short-form social video (Instagram Reel / TikTok). "
    "Watch it and return JSON with:\n"
    "- spoken_hook: the FIRST line actually spoken or shown in the first ~3 "
    "seconds, verbatim (original language). Empty string if there is no hook.\n"
    "- transcript: full speech-to-text of the video (original language). Empty "
    "if there is no speech.\n"
    "- on_screen_text: text overlays / captions burned into the video, joined "
    'with " / ". Empty if none.\n'
    "- format: the short-form format, e.g. \"talking head + text overlay\", "
    "\"photo slideshow\", \"green screen\", \"b-roll + voiceover\", \"screen "
    "recording demo\", \"POV skit\".\n"
    "- visual_tags: 3-6 short English tags for what is on screen (e.g. "
    "\"office\", \"laptop\", \"person to camera\", \"resume document\").\n"
    "- duration_sec: integer length of the clip in seconds.\n"
    "- is_slideshow: true only if it is a static photo carousel, not a video.\n"
)


def analyse_video_file(
    path: str, caption: str = "", views: int = 0, video_id: str | None = None,
    url: str = "",
) -> Video:
    """Upload one local .mp4 to Gemini and turn the model's read of it into a
    canonical ingest.Video. `caption` / `views` are whatever the user typed in
    (both optional); everything else is extracted from the pixels/audio.
    `url` is the clip's own post URL, if the user has it - it does nothing on
    its own, but lets the caller later fetch and merge REAL engagement for
    this exact clip via merge_apify_engagement().

    Raises on any failure (no key, upload/processing error, timeout, bad JSON)
    so the caller can surface it - there is no offline path for "understand a
    video file".
    """
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required to analyse a video file")

    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    timeout = config.GEMINI_VIDEO_TIMEOUT
    vid = video_id or os.path.splitext(os.path.basename(path))[0]

    uploaded = genai.upload_file(path=path)
    try:
        deadline = time.time() + timeout
        while uploaded.state.name == "PROCESSING":
            if time.time() > deadline:
                raise TimeoutError(
                    f"Gemini took over {timeout}s to process the upload"
                )
            time.sleep(2)
            uploaded = genai.get_file(uploaded.name)
        if uploaded.state.name != "ACTIVE":
            raise RuntimeError(
                f"Gemini could not process the video (state={uploaded.state.name})"
            )

        model = genai.GenerativeModel(config.GEMINI_VIDEO_MODEL)
        gen_cfg = genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=_analysis_schema(),
        )

        def _do():
            return model.generate_content(
                [uploaded, _ANALYSIS_PROMPT],
                generation_config=gen_cfg,
                request_options={"timeout": timeout},
            )

        resp = _call_with_timeout(_do, timeout)
    finally:
        try:
            genai.delete_file(uploaded.name)
        except Exception:  # noqa: BLE001 - cleanup only
            pass

    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    data = json.loads(text)

    return Video(
        id=str(vid),
        url=url.strip(),
        caption=caption.strip(),
        creator="own_post",
        transcript=str(data.get("transcript", "")).strip(),
        spoken_hook=str(data.get("spoken_hook", "")).strip(),
        detected_format=str(data.get("format", "")).strip(),
        on_screen_text=str(data.get("on_screen_text", "")).strip(),
        visual_tags=[str(x).strip() for x in data.get("visual_tags", []) if str(x).strip()],
        duration_sec=int(data.get("duration_sec") or 0),
        is_slideshow=bool(data.get("is_slideshow")),
        engagement={"views": int(views)} if views else {},
    )


def deepen_with_video_content(
    own_videos: list[Video], limit: int = 10, use_gemini: bool = False,
) -> tuple[list[Video], int, list[str]]:
    """For up to `limit` clips pulled in bulk via the Apify API (handle-based
    fetch), download the actual video file from its Apify-supplied
    `video_download_url` and read it for real (spoken hook, on-screen text,
    format) - so a whole account can be deep-analysed by handle instead of
    picking .mp4s by hand one at a time.

    `use_gemini=False` (default): reads locally via video_local - free,
    offline, no daily-quota risk, but a mechanical detected_format and no
    visual_tags (see video_local's docstring for the trade-off). `limit` then
    just bounds wall-clock time, not spend.
    `use_gemini=True`: reads via Gemini's File API (analyse_video_file) -
    also gets visual_tags and a genuinely judged format, but 1 Gemini call
    PER clip, so `limit` is really a spend cap here.

    Either way: skips photo/slideshow posts (nothing to watch) and any clip
    with no `video_download_url`. The candidates are the `limit` clips with
    the most REAL views (Apify engagement) - the account's current top
    performers - so the bounded budget (calls or just time) goes toward
    material worth naming specifically in "what works" / the portfolio
    rollup, rather than whatever order Apify happened to return. A clip with
    no view count (engagement missing) sorts last. Apify-sourced fields
    (engagement, sound, timestamp, caption) are kept as-is; only the
    pixel-derived fields are filled in. A clip whose download/analysis fails
    keeps its original caption-only data and is reported in `errors` rather
    than sinking the whole batch.

    Returns (videos, n_deepened, errors), `videos` in the same order/length
    as `own_videos`.
    """
    import tempfile

    import requests

    if use_gemini:
        reader = analyse_video_file
    else:
        from video_local import analyse_video_file_locally as reader

    eligible = [
        v for v in own_videos if v.video_download_url and not v.is_slideshow
    ]
    eligible.sort(key=lambda v: v.engagement.get("views", 0), reverse=True)
    candidates = eligible[:max(0, limit)]

    deepened: dict[str, Video] = {}
    errors: list[str] = []
    for v in candidates:
        tmp_path = None
        try:
            resp = requests.get(v.video_download_url, timeout=60)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            read = reader(tmp_path, caption=v.caption, video_id=v.id, url=v.url)
            deepened[v.id] = replace(
                v,
                transcript=read.transcript,
                spoken_hook=read.spoken_hook,
                detected_format=read.detected_format,
                on_screen_text=read.on_screen_text,
                visual_tags=read.visual_tags,
                duration_sec=read.duration_sec or v.duration_sec,
            )
        except Exception as e:  # noqa: BLE001 - one clip's failure shouldn't sink the batch
            errors.append(f"{v.id}: {type(e).__name__}: {e}")
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    result = [deepened.get(v.id, v) for v in own_videos]
    return result, len(deepened), errors


# --- Step 2: own videos vs the reference sheet ----------------------------

def _audit_schema():
    import google.generativeai as genai
    S, T = genai.protos.Schema, genai.protos.Type
    rewrite = S(type=T.OBJECT, properties={
        "original": S(type=T.STRING),
        "rewrite": S(type=T.STRING),
        "based_on": S(type=T.STRING),
    }, required=["original", "rewrite", "based_on"])
    strings = S(type=T.ARRAY, items=S(type=T.STRING))
    item = S(type=T.OBJECT, properties={
        "video_id": S(type=T.STRING),
        "assessment": S(type=T.STRING),
        "what_works": strings,
        "gaps": strings,
        "hook_rewrites": S(type=T.ARRAY, items=rewrite),
        "format_suggestion": S(type=T.STRING),
        "vocabulary_to_adopt": strings,
        "priority": S(type=T.STRING),
        "caveat": S(type=T.STRING),
    }, required=[
        "video_id", "assessment", "what_works", "gaps", "hook_rewrites",
        "format_suggestion", "vocabulary_to_adopt", "priority", "caveat",
    ])
    return S(type=T.ARRAY, items=item)


def _own_clip_block(v: Video) -> str:
    parts = [f"video_id: {v.id}"]
    if v.spoken_hook:
        parts.append(f"spoken hook: {v.spoken_hook}")
    if v.detected_format:
        parts.append(f"format: {v.detected_format}")
    if v.caption:
        parts.append(f"caption: {v.caption}")
    if v.on_screen_text:
        parts.append(f"on-screen text: {v.on_screen_text}")
    if v.transcript:
        tr = v.transcript if len(v.transcript) <= 700 else v.transcript[:700] + "…"
        parts.append(f"transcript: {tr}")
    if v.visual_tags:
        parts.append("visuals: " + ", ".join(v.visual_tags))
    if v.duration_sec:
        parts.append(f"duration: {v.duration_sec}s")
    metrics = ", ".join(f"{k}: {n:,}" for k, n in v.engagement.items() if n)
    if metrics:
        label = "REAL engagement (Apify)" if _has_real_engagement(v) else "views (user-supplied estimate)"
        parts.append(f"{label}: {metrics}")
    return "\n".join(parts)


def _norm_audit_item(raw: dict, language: str = "ja", has_engagement: bool = False) -> dict:
    def _lst(key):
        val = raw.get(key, [])
        if isinstance(val, str):
            val = [val]
        return [str(x).strip() for x in val if str(x).strip()] if isinstance(val, list) else []

    rewrites = []
    for rw in raw.get("hook_rewrites", []) or []:
        if not isinstance(rw, dict):
            continue
        orig = str(rw.get("original", "")).strip()
        new = str(rw.get("rewrite", "")).strip()
        if new:
            rewrites.append({
                "original": orig,
                "rewrite": new,
                "based_on": str(rw.get("based_on", "")).strip(),
            })

    prio = str(raw.get("priority", "")).strip().lower()
    if prio not in ("high", "medium", "low"):
        prio = "medium"

    return {
        "video_id": str(raw.get("video_id", "")).strip(),
        "assessment": str(raw.get("assessment", "")).strip(),
        "what_works": _lst("what_works"),
        "gaps": _lst("gaps"),
        "hook_rewrites": rewrites,
        "format_suggestion": str(raw.get("format_suggestion", "")).strip(),
        "vocabulary_to_adopt": _lst("vocabulary_to_adopt"),
        "priority": prio,
        "caveat": str(raw.get("caveat", "")).strip() or caveat_for(has_engagement, language),
        "source": "gemini",
    }


def _audit_fallback(
    own_videos: list[Video], ref: dict, language: str = "ja"
) -> list[dict]:
    """No-key audit: pure vocabulary set-difference + the reference hook list."""
    ref_vocab = ref.get("vocabulary", [])
    fb = _FB.get(language, _FB["ja"])
    out = []
    for v in own_videos:
        blob = " ".join(
            t for t in (v.spoken_hook, v.caption, v.on_screen_text, v.transcript)
            if t
        ).lower()
        present = [w for w in ref_vocab if w.lower().lstrip("#") in blob]
        missing = [w for w in ref_vocab if w.lower().lstrip("#") not in blob]
        out.append({
            "video_id": v.id,
            "assessment": "",
            "what_works": (
                [fb["matched"].format(v=", ".join(present))] if present else []
            ),
            "gaps": (
                [fb["missing"].format(v=", ".join(missing[:12]))] if missing else []
            ),
            "hook_rewrites": [],
            "format_suggestion": (
                fb["ref_formats"].format(v=", ".join(ref.get("formats", [])[:6]))
                if ref.get("formats") else ""
            ),
            "vocabulary_to_adopt": missing[:12],
            "priority": "medium",
            "caveat": caveat_for(_has_real_engagement(v), language),
            "source": "fallback",
            "needs_gemini_note": _NEEDS_GEMINI.get(language, _NEEDS_GEMINI["ja"]),
            "reference_hooks": ref.get("hook_patterns", [])[:8],
        })
    return out


def audit_against_reference(
    own_videos: list[Video], report: list[dict], brand: dict, language: str = "ja"
) -> list[dict]:
    """ONE Gemini call: compare every own clip to the niche pattern sheet built
    from `report`. Falls back to the mechanical set-difference on no key / any
    failure. Returns one audit dict per clip, in input order.
    """
    ref = reference_patterns(report)
    lang_name = LANG_NAME.get(language, "Japanese")

    if not own_videos:
        return []
    if not config.GEMINI_API_KEY:
        return _audit_fallback(own_videos, ref, language)

    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)

    tools = "\n".join(f"- {t}" for t in brand.get("tools", []))
    ref_hooks = "\n".join(
        f'  - {h["pattern"]}' + (f'  (e.g. "{h["example"]}")' if h.get("example") else "")
        for h in ref.get("hook_patterns", [])
    ) or "  (none extracted)"
    ref_text = (
        f"Niche themes: {', '.join(ref.get('themes', [])) or '(none)'}\n"
        f"Niche hook templates ACTUALLY used:\n{ref_hooks}\n"
        f"Niche formats: {', '.join(ref.get('formats', [])) or '(none)'}\n"
        f"Niche vocabulary / phrasings: {', '.join(ref.get('vocabulary', [])) or '(none)'}\n"
        f"Niche emotional tones: {', '.join(ref.get('emotional_tone', [])) or '(none)'}"
    )
    clips_text = "\n\n".join(f"### Clip\n{_own_clip_block(v)}" for v in own_videos)
    engagement_present = any(_has_real_engagement(v) for v in own_videos)

    if engagement_present:
        honesty_rule = (
            "IMPORTANT HONESTY RULE: some clips below carry REAL engagement "
            "(views / likes / comments / saves), pulled live from the brand's "
            "own account via the Apify API - marked 'REAL engagement (Apify)'. "
            "You MAY use it to say a clip under- or over-performed RELATIVE TO "
            "THE BRAND'S OTHER CLIPS in this batch, and weigh `priority` "
            "accordingly (a low-performer that also has gaps vs the niche is a "
            "stronger candidate for a rewrite than a clip already beating the "
            "batch). Clips marked 'views (user-supplied estimate)' or with no "
            "figure at all have no reliable number - never invent one, and "
            "never compare those clips by view count. Either way this is a "
            "snapshot (not a time series) and the niche reference is a "
            "hashtag-search snapshot too, so NEVER promise a change will go "
            "viral, trend, or beat a specific future number - only that it "
            "moves the post toward phrasing / formats the niche uses, or "
            f"toward the brand's own better-performing clips. Each item's "
            f"`caveat` must say this, in {lang_name}.\n\n"
        )
    else:
        honesty_rule = (
            "IMPORTANT HONESTY RULE: the reference below is a hashtag-search "
            "snapshot (not a time series) and the brand's own clips carry no "
            "reliable engagement. So NEVER promise a change will go viral, trend, "
            "or perform better. Frame every suggestion as 'moves this post toward "
            f"phrasing / formats the niche actually uses'. Each item's `caveat` "
            f"must say this, in {lang_name}.\n\n"
        )

    prompt = (
        "You are a short-form social strategist reviewing a brand's OWN "
        "Instagram Reels / TikToks against what the job-hunt / career niche is "
        "currently doing.\n\n"
        + honesty_rule +
        f"OUTPUT LANGUAGE: write assessment, what_works, gaps, "
        f"format_suggestion, caveat and hook_rewrites.rewrite in {lang_name}, "
        f"EVEN IF the niche reference below is written in another language. "
        f"Keep quoted phrasings (vocabulary_to_adopt, and the clip's original "
        f"hook) in their original language.\n\n"
        f"Brand: {brand.get('name', '')}\n"
        f"Brand description: {brand.get('description', '')}\n"
        f"Target audience: {brand.get('target_audience', '')}\n"
        f"Brand tone: {brand.get('tone', '')}\n"
        f"Brand tools (use EXACT names):\n{tools}\n\n"
        f"--- NICHE REFERENCE (what others do) ---\n{ref_text}\n\n"
        f"--- THE BRAND'S OWN CLIPS TO REVIEW ---\n{clips_text}\n\n"
        "For EACH clip return an object with:\n"
        "- video_id: the clip's video_id, verbatim\n"
        f"- assessment: 1-2 sentences ({lang_name}), overall read of the clip's "
        "hook + format vs the niche\n"
        f"- what_works: things the clip already does that match the niche "
        f"({lang_name}, may be empty)\n"
        "- gaps: concrete things missing vs the niche - weak/slow hook, format "
        f"not used here, vocabulary absent ({lang_name})\n"
        "- hook_rewrites: 1-3 rewrites of THIS clip's opening line ONLY. This "
        "is a rewording of the SAME footage/topic the clip already shows - "
        "NOT a new video concept. Do not swap in a different subject, "
        "scenario, or claim that isn't actually in this clip's footage, "
        "caption, or transcript (e.g. if the clip is a founder/event/office "
        "vlog with no resume/interview content, do NOT rewrite it into an "
        "interview-tips or resume-tips hook - instead phrase what the clip "
        "ACTUALLY shows using niche hook energy/structure, or state in "
        "`gaps` that this clip's premise has no natural tie-in and a hook "
        "rewrite would misrepresent it). NEVER invent an event, achievement, "
        "quote, statistic, or piece of dialogue that isn't in the clip's "
        "actual footage/caption/transcript just to manufacture a tie-in - e.g. "
        "for a founding-team or office clip, do NOT write a rewrite implying "
        "they \"analyzed thousands of ATS rejections\" or \"built a tool to "
        "beat ATS filters\" or invent a line of dialogue between coworkers "
        "about resumes if that never happened on screen. A rewrite may only "
        "resequence/re-emphasize words and claims that are ALREADY present in "
        "the clip's own hook/caption/transcript/on-screen text - it must not "
        "add any new factual claim, number, or scene. If the clip has nothing "
        "usable to build a truthful rewrite from, leave hook_rewrites empty "
        "and explain the gap instead. Each = {\"original\": the clip's "
        "current hook, \"rewrite\": a punchier version of the SAME content "
        "built on a niche hook template, ready to film (keep the clip's own "
        "topic/subject matter/language - only change phrasing/structure, "
        "never the underlying claim or scenario), \"based_on\": which niche "
        "template/phrasing it uses}\n"
        f"- format_suggestion: one {lang_name} sentence - the niche format this "
        "clip should try, or '' if its format is already fine\n"
        "- vocabulary_to_adopt: niche phrasings this clip could naturally use "
        "(keep original language, may be empty)\n"
        "- priority: \"high\" / \"medium\" / \"low\" - how far this clip is from "
        "the niche" + (
            ", factoring in REAL engagement where present (a low-performer "
            "with niche gaps ranks higher priority than a clip already "
            "outperforming the batch)" if engagement_present else ""
        ) + "\n"
        f"- caveat: the {lang_name} honesty disclaimer above\n\n"
        "Return a JSON array, one object per clip, video_id matching."
    )

    gen_cfg = genai.types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=_audit_schema(),
    )

    def _do():
        return model.generate_content(
            prompt,
            generation_config=gen_cfg,
            request_options={"timeout": config.GEMINI_TIMEOUT},
        )

    try:
        resp = _call_with_timeout(_do, config.GEMINI_TIMEOUT)
        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("["):]
        data = json.loads(text)
    except Exception:  # noqa: BLE001 - any failure -> mechanical fallback
        return _audit_fallback(own_videos, ref, language)

    by_id = {
        str(it.get("video_id", "")).strip(): it
        for it in (data if isinstance(data, list) else [])
        if isinstance(it, dict)
    }
    return [
        _norm_audit_item(
            by_id.get(v.id, {"video_id": v.id}), language, _has_real_engagement(v),
        )
        for v in own_videos
    ]


# --- Step 3: portfolio rollup (the "scoreboard") --------------------------

def _portfolio_schema():
    import google.generativeai as genai
    S, T = genai.protos.Schema, genai.protos.Type
    return S(type=T.OBJECT, properties={
        "overall": S(type=T.STRING),
        "recommendations": S(type=T.ARRAY, items=S(type=T.STRING)),
    }, required=["overall", "recommendations"])


def _engagement_stats(own_videos: list[Video]) -> dict:
    """Portfolio-level read of REAL (Apify) view counts, if any clip has one:
    median + which clip is the current best / worst performer by views. Empty
    dict when no clip carries real engagement."""
    rows = [
        (v.id, v.engagement["views"])
        for v in own_videos
        if _has_real_engagement(v) and v.engagement.get("views")
    ]
    if not rows:
        return {}
    rows.sort(key=lambda r: r[1])
    values = [n for _, n in rows]
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return {
        "n_with_engagement": len(rows),
        "median_views": int(median),
        "lowest": {"video_id": rows[0][0], "views": rows[0][1]},
        "highest": {"video_id": rows[-1][0], "views": rows[-1][1]},
    }


_SUMM_FB = {
    "ja": {
        "vocab": "複数投稿で共通して不足している語彙を横断的に採用: {v}",
        "high": "優先度『高』の {n} 本を先に作り直す。",
    },
    "en": {
        "vocab": "Adopt the niche vocabulary missing across several clips: {v}",
        "high": "Rework the {n} high-priority clip(s) first.",
    },
}


def summarise_audit(
    audit_items: list[dict],
    own_videos: list[Video],
    report: list[dict],
    brand: dict,
    language: str = "ja",
) -> dict:
    """Roll the per-clip audits up into one scoreboard:

    {
      "n_clips", "priority_counts": {high, medium, low},
      "top_vocabulary_gaps": [{"term", "count"}],   # phrasing suggested across clips
      "sound": {...}  from sound_trends.own_sound_summary,
      "engagement": {} or {"n_with_engagement", "median_views", "lowest", "highest"},
      "overall": str,               # Gemini paragraph ('' on fallback)
      "recommendations": [str],      # cross-cutting, actionable
      "caveat": str,
      "source": "gemini" | "fallback",
    }
    """
    n = len(audit_items)
    pri = Counter(it.get("priority", "medium") for it in audit_items)
    vocab = Counter(
        term for it in audit_items for term in it.get("vocabulary_to_adopt", [])
    )
    top_vocab = [{"term": t, "count": c} for t, c in vocab.most_common(12)]
    sound = sound_trends.own_sound_summary(own_videos)
    engagement = _engagement_stats(own_videos)

    sfb = _SUMM_FB.get(language, _SUMM_FB["ja"])
    base = {
        "n_clips": n,
        "priority_counts": {k: pri.get(k, 0) for k in ("high", "medium", "low")},
        "top_vocabulary_gaps": top_vocab,
        "sound": sound,
        "engagement": engagement,
        "caveat": caveat_for(bool(engagement), language),
    }

    if n < 2 or not config.GEMINI_API_KEY:
        recs = []
        if top_vocab:
            recs.append(sfb["vocab"].format(
                v=", ".join(x["term"] for x in top_vocab[:8])
            ))
        if pri.get("high"):
            recs.append(sfb["high"].format(n=pri["high"]))
        return {**base, "overall": "", "recommendations": recs, "source": "fallback"}

    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    ref = reference_patterns(report)
    lang_name = LANG_NAME.get(language, "Japanese")

    own_by_id = {v.id: v for v in own_videos}

    def _views_note(video_id: str) -> str:
        v = own_by_id.get(video_id)
        if v is None or not _has_real_engagement(v) or not v.engagement.get("views"):
            return ""
        return f"  views (REAL, Apify): {v.engagement['views']:,}\n"

    per_clip = "\n\n".join(
        f"clip {it.get('video_id', '?')} (priority {it.get('priority', '?')}):\n"
        f"{_views_note(it.get('video_id', ''))}"
        f"  assessment: {it.get('assessment', '')}\n"
        f"  gaps: {'; '.join(it.get('gaps', [])) or '-'}\n"
        f"  vocab to adopt: {', '.join(it.get('vocabulary_to_adopt', [])) or '-'}"
        for it in audit_items
    )
    honesty_line = (
        "HONESTY RULE: some clips carry REAL engagement (views) from the "
        f"brand's own account (median across this batch: {engagement['median_views']:,}"
        f" views). You MAY name specific under/over-performers by id and use "
        "that to prioritise recommendations, but the reference is still a "
        "snapshot - never promise virality or a specific future number; frame "
        "as moving the account toward niche phrasing/formats and toward the "
        "brand's own better-performing clips.\n\n"
        if engagement else
        "HONESTY RULE: no reliable engagement, reference is a snapshot - never "
        "promise virality or better performance; frame as moving the whole "
        "account toward niche phrasing/formats.\n\n"
    )
    prompt = (
        "You reviewed several of a brand's own short-form clips against a "
        "job-hunt / career niche reference. Below are the per-clip findings. "
        "Write a PORTFOLIO-level read.\n\n"
        + honesty_line +
        f"Brand: {brand.get('name', '')} - {brand.get('description', '')}\n"
        f"Niche hook templates: {', '.join(h['pattern'] for h in ref.get('hook_patterns', [])) or '-'}\n"
        f"Niche formats: {', '.join(ref.get('formats', [])) or '-'}\n"
        f"Niche vocabulary: {', '.join(ref.get('vocabulary', [])) or '-'}\n\n"
        f"--- PER-CLIP FINDINGS ({n} clips) ---\n{per_clip}\n\n"
        f"Write in {lang_name}, even if the findings above are in another "
        f"language. Return JSON: {{\"overall\": a 2-4 sentence {lang_name} "
        "paragraph naming the recurring pattern across clips (e.g. hooks "
        "consistently slower than the niche, an angle never used), "
        f"\"recommendations\": 3-5 {lang_name} bullets, each a concrete "
        "cross-cutting action for the next batch of posts}."
    )
    gen_cfg = genai.types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=_portfolio_schema(),
    )

    def _do():
        return model.generate_content(
            prompt, generation_config=gen_cfg,
            request_options={"timeout": config.GEMINI_TIMEOUT},
        )

    try:
        resp = _call_with_timeout(_do, config.GEMINI_TIMEOUT)
        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):]
        data = json.loads(text)
        overall = str(data.get("overall", "")).strip()
        recs = [str(x).strip() for x in data.get("recommendations", []) if str(x).strip()]
        return {**base, "overall": overall, "recommendations": recs, "source": "gemini"}
    except Exception:  # noqa: BLE001 - fall back to the mechanical rollup
        recs = []
        if top_vocab:
            recs.append(sfb["vocab"].format(
                v=", ".join(x["term"] for x in top_vocab[:8])
            ))
        return {**base, "overall": "", "recommendations": recs, "source": "fallback"}
