"""
Ingestion layer.

This is the ONLY part of the system that knows where data comes from.
Today it reads a static JSON file. Tomorrow you can write a new loader
(official API, Bright Data, Apify, a compliant crawler) that returns the
same list[Video], and nothing downstream has to change.

That is the "keep the ingestion layer separate / swappable" instruction,
made concrete.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Video:
    """Canonical shape of one social video, independent of its source.

    Any new data source must produce objects of this shape. Fields we can't
    get yet (e.g. transcript from a real video) can be filled in by a later
    processing step; in Stage 1a they already come with the sample data.
    """
    id: str
    url: str = ""
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    creator: str = ""
    timestamp: str = ""                       # ISO 8601 string
    engagement: dict[str, int] = field(default_factory=dict)
    engagement_is_real: bool = False           # True only for a live scrape (Apify); False for a user-typed estimate
    video_download_url: str = ""               # direct CDN link to the video file, if the scrape exposed one (self-audit deep-analysis)
    sound_id: str = ""
    sound_name: str = ""
    sound_is_original: bool = False           # creator's own audio vs a reused/licensed sound
    transcript: str = ""                       # speech -> text (Whisper later)
    spoken_hook: str = ""                       # first spoken/shown line (video_audit)
    detected_format: str = ""                   # short-form format read off the video (video_audit)
    on_screen_text: str = ""                   # OCR text overlays
    visual_tags: list[str] = field(default_factory=list)
    duration_sec: int = 0                      # clip length; 0 = unknown
    is_slideshow: bool = False                 # photo-carousel post, not a video

    def analysis_text(self) -> str:
        """The single text blob we feed to the embedding model.

        We deliberately fuse several signals (caption, on-screen text,
        transcript, hashtags) because on any one of them alone the picture is
        partial. Hashtags are repeated lightly as they are strong topic hints.
        """
        parts = [
            self.caption,
            self.on_screen_text,
            self.transcript,
            " ".join(f"#{h}" for h in self.hashtags),
        ]
        return "  ".join(p for p in parts if p).strip()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_videos(path: str) -> list[Video]:
    """Load videos from a local JSON file into the canonical Video shape."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    videos: list[Video] = []
    for item in raw:
        # Only pass keys the dataclass knows about, so extra fields in the
        # source don't crash the loader.
        known = {k: item[k] for k in Video.__dataclass_fields__ if k in item}
        videos.append(Video(**known))
    return videos


def _clean_str(value: Any) -> str:
    """Normalize a raw field to a plain string, treating None / the literal
    string "null" (as some HF datasets encode missing values) as empty."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "null" else text


_MOJIBAKE_MARKERS = ("Ã", "â\x80", "â\x82", "ð\x9f", "Â\xa0", "â\x9d", "â\x9c")


def _fix_mojibake(text: str) -> str:
    """Best-effort repair of UTF-8 that was decoded as Latin-1 by the scraper
    (e.g. "donâ€™t" -> "don't", garbled emoji -> real emoji).

    Only attempts the round-trip when the string actually looks mangled, and
    falls back to the original on any failure so clean text is never corrupted.
    """
    if not text or not any(m in text for m in _MOJIBAKE_MARKERS):
        return text
    try:
        return text.encode("latin-1", "strict").decode("utf-8", "strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def load_videos_from_huggingface(limit: int = 1000) -> list[Video]:
    """Load real TikTok videos from the Hugging Face dataset
    lingbow/tiktok-video-engagement-200k, streamed so we only pull `limit`
    rows instead of downloading the whole ~200k-row dataset.

    Same swap-in point as load_videos(): this returns the same canonical
    list[Video], so nothing downstream (embedding, clustering, explaining,
    brand matching) needs to know or care where the videos came from.

    Filters to is_english == 1 rows, and drops rows with no caption AND no
    transcript (nothing for the embedding/explanation layers to work with).
    engagement is left as {} - this dataset doesn't carry view/like counts.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "lingbow/tiktok-video-engagement-200k", split="train", streaming=True
    )

    videos: list[Video] = []
    for row in ds:
        if row.get("is_english") != 1:
            continue

        caption = _clean_str(row.get("desc"))
        transcript = _clean_str(row.get("transcript"))
        if not caption and not transcript:
            continue

        hashtags = [
            h.get("hashtag_name", "")
            for h in (row.get("hashtags") or [])
            if h.get("hashtag_name")
        ]

        videos.append(Video(
            id=str(row.get("video_id", "")),
            caption=caption,
            hashtags=hashtags,
            creator=str(row.get("author_id", "")),
            timestamp=str(row.get("create_time", "")),
            sound_id=str(row.get("music_id", "")),
            sound_name=_clean_str(row.get("music_title")),
            transcript=transcript,
            on_screen_text=_clean_str(row.get("sticker_text")),
        ))

        if len(videos) >= limit:
            break

    return videos


_LATIN_LETTERS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "àáâãäåèéêëìíîïòóôõöùúûüñçýÿ"
)


def _latin_ratio(text: str) -> float:
    """Share of alphabetic characters that are plain/accented Latin letters.
    Used only as a fallback when langdetect isn't installed or can't decide -
    good enough to catch Arabic/Cyrillic/CJK/Devanagari script text, not
    meant to distinguish English from other Latin-script languages."""
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 1.0
    latin = sum(1 for c in letters if c in _LATIN_LETTERS)
    return latin / len(letters)


def _is_english_text(text: str) -> bool:
    """Best-effort language check for one caption/transcript blob.

    Too little text to judge (empty caption, an emoji-only line, a couple of
    hashtags) is kept rather than dropped - there is nothing to misjudge, and
    an over-eager filter would silently shrink an otherwise on-topic cluster
    for no signal.
    """
    text = text.strip()
    if len(text) < 8:
        return True
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0  # deterministic across runs
        return detect(text) == "en"
    except Exception:
        return _latin_ratio(text) > 0.85


def filter_english(videos: list[Video]) -> tuple[list[Video], int]:
    """Keep only videos whose caption/transcript/on-screen text reads as
    English, dropping the rest. Returns (kept, n_dropped).

    Applied to every scrape source (a hashtag search pulls in whatever
    language happens to reuse the same tag - #jobsearch shows up under Urdu,
    Arabic-script and stylised-Unicode captions too), but not to
    load_videos_from_huggingface, which already filters on that dataset's own
    is_english column.
    """
    kept: list[Video] = []
    dropped = 0
    for v in videos:
        text = " ".join(t for t in (v.caption, v.transcript, v.on_screen_text) if t)
        if _is_english_text(text):
            kept.append(v)
        else:
            dropped += 1
    return kept, dropped


# --- Spam / low-quality caption filter -----------------------------------------
#
# A different problem from language: keyword-stuffed / decorative-glyph spam
# (e.g. "ワ爪ル八ロロ / tohelpyo / ExPloretemplates / pccni rendcd / …") reads as
# English to langdetect - it's mostly Latin-letter tokens - so filter_english()
# lets it through. What actually marks it as junk is that it (a) sprinkles in
# characters from a script (CJK / Kana / Hangul / Arabic / Devanagari /
# Cyrillic) that has no business appearing in an English caption, or (b) is
# dominated by immediate character repetition (keyboard-mash / emoji-mash
# spam like "aaaaaa" or "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥") rather than real words.

_FOREIGN_SCRIPT_RANGES = (
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0x3400, 0x4DBF),   # CJK extension A
    (0x4E00, 0x9FFF),   # CJK unified ideographs
    (0xAC00, 0xD7A3),   # Hangul syllables
    (0x0600, 0x06FF),   # Arabic
    (0x0900, 0x097F),   # Devanagari
    (0x0400, 0x04FF),   # Cyrillic
)
_DECORATIVE_CJK_PUNCT = "『』「」【】〜・"


def _is_foreign_script_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _FOREIGN_SCRIPT_RANGES)


def _foreign_glyph_ratio(text: str) -> float:
    """Share of alphabetic characters drawn from a non-Latin script.

    A caption that's genuinely in another language is caught by
    filter_english() first (this ratio would be high, near 1.0). What this
    catches is the leftover case: a handful of such characters sprinkled into
    an otherwise Latin/English caption - not a language, decoration."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    foreign = sum(1 for c in letters if _is_foreign_script_char(c))
    return foreign / len(letters)


def _max_char_run(text: str) -> int:
    """Length of the longest run of one repeated (non-space) character."""
    run = maxrun = 0
    prev = None
    for c in text:
        if c.isspace():
            prev = None
            run = 0
            continue
        if c == prev:
            run += 1
        else:
            run = 1
            prev = c
        maxrun = max(maxrun, run)
    return maxrun


def _is_spam_text(text: str) -> bool:
    """Best-effort junk check for one caption/transcript blob.

    Too little text is kept, same reasoning as _is_english_text(): nothing to
    misjudge, so an empty/very short caption is never flagged as spam here.
    """
    text = text.strip()
    if len(text) < 8:
        return False
    foreign_chars = sum(1 for c in text if c.isalpha() and _is_foreign_script_char(c))
    decorative = sum(1 for c in text if c in _DECORATIVE_CJK_PUNCT)
    if foreign_chars >= 2 and _foreign_glyph_ratio(text) < 0.5:
        return True
    if decorative >= 2:
        return True
    if _max_char_run(text) >= 6:
        return True
    return False


def filter_low_quality(videos: list[Video]) -> tuple[list[Video], int]:
    """Drop videos whose caption/transcript/on-screen text looks like
    keyword-stuffed or decorative-glyph spam rather than a real post. Returns
    (kept, n_dropped). Meant to run after filter_english(), on what's left."""
    kept: list[Video] = []
    dropped = 0
    for v in videos:
        text = " ".join(t for t in (v.caption, v.transcript, v.on_screen_text) if t)
        if _is_spam_text(text):
            dropped += 1
        else:
            kept.append(v)
    return kept, dropped


def load_videos_from_apify(path: str) -> list[Video]:
    """Load videos scraped by the Apify TikTok Scraper (clockworks/tiktok-scraper
    output shape). Same swap-in point as load_videos(): returns the canonical
    list[Video], so nothing downstream needs to know the source.

    `path` may be a single JSON file OR a directory of them (one file per scrape
    run / search term). Files are merged and de-duplicated by video id, because
    the same clip is returned under several `searchHashtag` inputs.

    Unlike the sample data, this source DOES carry engagement, so `engagement`
    is populated from playCount / diggCount / commentCount / shareCount /
    collectCount. It is still a keyword-search snapshot (not a time series), so
    downstream treats those counts as a relative, un-normalised signal only.

    Transcripts are not inlined by the scraper (only downloadable subtitle
    files), and there is no OCR, so `transcript` and `on_screen_text` stay empty
    - caption + hashtags carry the analysis.
    """
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.lower().endswith(".json")
        )
    else:
        files = [path]

    seen: set[str] = set()
    videos: list[Video] = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for item in raw:
            v = _video_from_apify_tiktok(item)
            if v is None or v.id in seen:
                continue
            seen.add(v.id)
            videos.append(v)

    videos, n_dropped = filter_english(videos)
    if n_dropped:
        print(f"[lang-filter] dropped {n_dropped} non-English video(s), "
              f"{len(videos)} remaining")
    videos, n_spam = filter_low_quality(videos)
    if n_spam:
        print(f"[spam-filter] dropped {n_spam} low-quality/spam video(s), "
              f"{len(videos)} remaining")
    return videos


# --- Apify row -> Video (shared by the file loader and the API loader) -------

def _engagement_metric(*raws: Any) -> int | None:
    """First present raw metric, or None if every candidate is missing or
    negative. Instagram returns -1 (not 0, not absent) for a like/view count
    the creator has hidden - that is never a real count, so it must not be
    stored as one (it would render as a nonsensical "-1" or, worse, silently
    become a false "0")."""
    for raw in raws:
        if raw is None:
            continue
        try:
            n = int(raw)
        except (TypeError, ValueError):
            continue
        if n >= 0:
            return n
    return None


def _video_from_apify_tiktok(item: dict) -> Video | None:
    """Map one clockworks/tiktok-scraper row to a Video, or None if it carries
    no id / no text to analyse."""
    vid = str(item.get("id", "")).strip()
    caption = _fix_mojibake(_clean_str(item.get("text")))
    hashtags = [
        h["name"].strip()
        for h in (item.get("hashtags") or [])
        if isinstance(h, dict) and h.get("name") and h["name"].strip()
    ]
    if not vid or (not caption and not hashtags):
        return None

    author = item.get("authorMeta") or {}
    music = item.get("musicMeta") or {}
    vmeta = item.get("videoMeta") or {}
    engagement = {}
    for key, raw in (
        ("views", item.get("playCount")), ("likes", item.get("diggCount")),
        ("comments", item.get("commentCount")), ("shares", item.get("shareCount")),
        ("saves", item.get("collectCount")),
    ):
        n = _engagement_metric(raw)
        if n is not None:
            engagement[key] = n
    return Video(
        id=vid,
        url=str(item.get("webVideoUrl", "")),
        caption=caption,
        hashtags=hashtags,
        creator=str(author.get("name", "")),
        timestamp=str(item.get("createTimeISO", "")),
        duration_sec=int(vmeta.get("duration") or 0),
        is_slideshow=bool(item.get("isSlideshow")),
        engagement=engagement,
        engagement_is_real=True,
        sound_id=str(music.get("musicId", "")),
        sound_name=_fix_mojibake(_clean_str(music.get("musicName"))),
        sound_is_original=bool(music.get("musicOriginal")),
    )


def _video_from_apify_instagram(item: dict) -> Video | None:
    """Map one apify/instagram-scraper post row to a Video, or None if unusable.

    Reels carry `musicInfo` (song_name / artist_name / audio_id /
    uses_original_audio); we keep audio_id + song_name in the same sound_id /
    sound_name fields the TikTok path uses, so "same sound reused across posts"
    stays detectable regardless of source.
    """
    pid = str(item.get("id") or item.get("shortCode") or "").strip()
    caption = _fix_mojibake(_clean_str(item.get("caption")))
    hashtags = [
        str(h).lstrip("#").strip()
        for h in (item.get("hashtags") or [])
        if str(h).strip()
    ]
    if not pid or (not caption and not hashtags):
        return None

    ptype = str(item.get("type") or "").lower()
    music = item.get("musicInfo") or {}
    engagement = {}
    views = _engagement_metric(item.get("videoViewCount"), item.get("videoPlayCount"))
    if views is not None:
        engagement["views"] = views
    for key, raw in (
        ("likes", item.get("likesCount")), ("comments", item.get("commentsCount")),
    ):
        n = _engagement_metric(raw)
        if n is not None:
            engagement[key] = n
    return Video(
        id=pid,
        url=str(item.get("url", "")),
        caption=caption,
        hashtags=hashtags,
        creator=str(item.get("ownerUsername", "")),
        timestamp=str(item.get("timestamp", "")),
        duration_sec=int(float(item.get("videoDuration") or 0)),
        is_slideshow=ptype in ("sidecar", "image"),
        engagement=engagement,
        video_download_url=str(item.get("videoUrl", "")),
        engagement_is_real=True,
        sound_id=str(music.get("audio_id", "")),
        sound_name=_clean_str(music.get("song_name")),
        sound_is_original=bool(music.get("uses_original_audio")),
    )


def load_videos_from_apify_api(
    *,
    platform: str,
    hashtags: list[str] | None = None,
    profiles: list[str] | None = None,
    post_urls: list[str] | None = None,
    results_limit: int = 100,
    token: str = "",
    tiktok_actor: str = "clockworks/tiktok-scraper",
    instagram_actor: str = "apify/instagram-scraper",
    sync_timeout: int = 240,
    reels_only: bool = False,
) -> list[Video]:
    """Run the Apify TikTok or Instagram scraper live (needs an API token) and
    return canonical Videos - the same shape as every other loader.

    Exactly one query kind is expected: `hashtags`, `profiles` (handles or
    profile URLs), or `post_urls`. Engagement is populated (this source has it),
    so downstream treats the counts as a relative, un-normalised signal.

    `reels_only` (Instagram only; ignored for TikTok, which has no photo
    posts to begin with): asks the actor for `resultsType: "reels"` instead
    of `"posts"`, so every row IS a video (never a photo/carousel slideshow)
    - every row is then eligible for video_audit.deepen_with_video_content()
    /video_local's actually-watch-it path, instead of some share of the batch
    coming back as slideshow posts with nothing to watch.
    """
    import apify_client

    hashtags = [h.lstrip("#").strip() for h in (hashtags or []) if h.strip()]
    profiles = [p.strip() for p in (profiles or []) if p.strip()]
    post_urls = [u.strip() for u in (post_urls or []) if u.strip()]
    if not (hashtags or profiles or post_urls):
        raise ValueError("give one of hashtags / profiles / post_urls")

    platform = platform.lower()
    if platform == "tiktok":
        run_input: dict = {
            "resultsPerPage": results_limit,
            "shouldDownloadVideos": False,
            "shouldDownloadCovers": False,
            "shouldDownloadSubtitles": False,
            "shouldDownloadSlideshowImages": False,
        }
        if hashtags:
            run_input["hashtags"] = hashtags
        if profiles:
            run_input["profiles"] = [p.lstrip("@") for p in profiles]
        if post_urls:
            run_input["postURLs"] = post_urls
        rows = apify_client.run_actor_sync(
            tiktok_actor, run_input, token, sync_timeout
        )
        mapper = _video_from_apify_tiktok

    elif platform == "instagram":
        if post_urls:
            urls = post_urls
        elif profiles:
            urls = [
                p if p.startswith("http")
                else f"https://www.instagram.com/{p.lstrip('@')}/"
                for p in profiles
            ]
        else:
            urls = [
                f"https://www.instagram.com/explore/tags/{h}/" for h in hashtags
            ]
        run_input = {
            "directUrls": urls,
            "resultsType": "reels" if reels_only else "posts",
            "resultsLimit": results_limit,
        }
        rows = apify_client.run_actor_sync(
            instagram_actor, run_input, token, sync_timeout
        )
        mapper = _video_from_apify_instagram

    else:
        raise ValueError(f"platform must be 'tiktok' or 'instagram', got {platform!r}")

    seen: set[str] = set()
    videos: list[Video] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        v = mapper(item)
        if v is None or v.id in seen:
            continue
        seen.add(v.id)
        videos.append(v)

    videos, n_dropped = filter_english(videos)
    if n_dropped:
        print(f"[lang-filter] dropped {n_dropped} non-English video(s), "
              f"{len(videos)} remaining")
    videos, n_spam = filter_low_quality(videos)
    if n_spam:
        print(f"[spam-filter] dropped {n_spam} low-quality/spam video(s), "
              f"{len(videos)} remaining")
    return videos
