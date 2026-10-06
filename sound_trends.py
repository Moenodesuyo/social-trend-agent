"""
Sound / audio signal layer.

The Apify scrapers already hand us a sound id per clip (TikTok `musicId`,
Instagram reels `audio_id`) plus an "is this the creator's own audio" flag
(`musicOriginal` / `uses_original_audio`). Nothing used it until now.

`analyse_sounds()` groups a video list by sound id and surfaces the sounds that
recur across SEVERAL creators - a cheap, no-API "this audio is being reused in
the niche" signal - plus how much of the niche rides original audio vs a
reused/licensed track.

Honesty guardrail (same as everywhere else): this is a hashtag-search snapshot
with cumulative, un-normalised view counts. "Reused across N posts in the
sample" is all it says - never "this sound is trending right now".
"""
from __future__ import annotations
from collections import Counter
from statistics import median

from ingest import Video

CAVEAT = (
    "ハッシュタグ検索の一時点スナップショットでの集計です。再生数は累計・"
    "フォロワー未補正なので、『このサウンドがいま伸びている』とは断定できません"
    "（『サンプル内で複数の投稿に使われている』まで）。"
)
CAVEAT_EN = (
    "Tallied from a one-shot hashtag-search snapshot. View counts are cumulative "
    "and not follower-normalised, so this is not evidence a sound is trending - "
    "only that it recurs across posts in the sample."
)


def _caveat(language: str) -> str:
    return CAVEAT_EN if language == "en" else CAVEAT


def _views(v: Video) -> int:
    return int(v.engagement.get("views") or 0)


def analyse_sounds(
    videos: list[Video],
    engagement_available: bool = False,
    min_creators: int = 2,
    top_n: int = 15,
    language: str = "ja",
) -> dict:
    """Group `videos` by sound id. Returns:

    {
      "sounds": [ ... one dict per reused sound, ranked ... ],
      "n_with_sound": how many clips carried a sound id,
      "original_share_pct": % of those on the creator's own audio (or None),
      "caveat": the snapshot disclaimer,
    }

    A sound is only listed if `min_creators` distinct creators used it - one
    account reusing its own audio is not a niche trend.
    """
    with_sound = [v for v in videos if v.sound_id]
    n = len(with_sound)

    by_id: dict[str, list[Video]] = {}
    for v in with_sound:
        by_id.setdefault(v.sound_id, []).append(v)

    sounds = []
    for sid, clips in by_id.items():
        creators = sorted({c.creator for c in clips if c.creator})
        if len(creators) < min_creators:
            continue
        vs = [_views(c) for c in clips if _views(c)]
        tags = Counter(h for c in clips for h in c.hashtags)
        top = sorted(clips, key=_views, reverse=True) if engagement_available else clips
        unknown = "(name unknown)" if language == "en" else "(名称不明)"
        sounds.append({
            "sound_id": sid,
            "sound_name": next((c.sound_name for c in clips if c.sound_name), "") or unknown,
            "is_original": any(c.sound_is_original for c in clips),
            "post_count": len(clips),
            "creator_count": len(creators),
            "creators": creators[:6],
            "median_views": int(median(vs)) if vs else None,
            "total_views": sum(vs) if vs else None,
            "hashtags": [h for h, _ in tags.most_common(6)],
            "examples": [
                {"caption": c.caption, "creator": c.creator, "url": c.url,
                 "views": _views(c) or None}
                for c in top[:3]
            ],
        })

    sounds.sort(
        key=lambda s: (s["post_count"], s["median_views"] or 0), reverse=True
    )

    n_original = sum(1 for v in with_sound if v.sound_is_original)
    return {
        "sounds": sounds[:top_n],
        "n_with_sound": n,
        "original_share_pct": round(100 * n_original / n) if n else None,
        "caveat": _caveat(language),
    }


def own_sound_summary(own_videos: list[Video]) -> dict:
    """Compact read of an account's OWN sound usage, for the self-audit
    scoreboard. Empty-ish when the clips carry no sound metadata (e.g. uploaded
    mp4s - Gemini doesn't return a sound id)."""
    with_sound = [v for v in own_videos if v.sound_id]
    n = len(with_sound)
    if not n:
        return {"n_with_sound": 0}
    n_original = sum(1 for v in with_sound if v.sound_is_original)
    reused = Counter(v.sound_id for v in with_sound if not v.sound_is_original)
    return {
        "n_with_sound": n,
        "original_share_pct": round(100 * n_original / n),
        "distinct_reused_sounds": len(reused),
    }
