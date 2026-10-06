"""
End-to-end pipeline orchestrator.

New purpose
-----------
The deliverable is MARKETING LANGUAGE INSIGHT for a brand's short-form
social, not a brand-relevance score. Flow:

    ingest -> fuse text -> embed -> cluster
           -> relevance gate (cheap, no-API): job-hunt/career clusters only
           -> language insight for those clusters (1 Gemini call total, or a
              mechanical vocabulary/theme fallback with no key)
           -> print a marketer-readable report

`brand_match` is kept but demoted: its score only gates which clusters get an
insight, so we never emit language advice for an unrelated cluster.

Data sources
------------
- json         : the bundled sample_videos.json (no engagement)
- huggingface  : lingbow/tiktok-video-engagement-200k (no engagement)
- apify        : Apify TikTok Scraper JSON in data/apify/ (HAS engagement:
                 playCount / diggCount / ...)

Honesty: the report's caveat matches the data. With no engagement it only says
a phrasing is "being used". With engagement it may compare phrasings by view
count WITHIN the sample, but flags that this is a keyword-search snapshot
(not a time series) with cumulative, un-normalised counts - never "going viral".

Every run also writes report.json (the structured list run() returns) and
report.md (human-readable) via exporter.py, unless --no-save.

`generate.py` overlaps with insight.py and is no longer called here; kept for
possible later use.

Run:
    python pipeline.py --backend tfidf
    python pipeline.py --backend tfidf --source apify --out out/
    python pipeline.py --backend tfidf --no-save
"""
from __future__ import annotations
import argparse
import sys
from statistics import median

import config
import exporter
from ingest import (
    load_videos,
    load_videos_from_huggingface,
    load_videos_from_apify,
    load_videos_from_apify_api,
)
from analyze import cluster_videos, distinctive_cluster_keywords, group_by_cluster
from explain import explain_fallback
from brand_match import (
    load_brand, gatekeeper_score, distinct_brand_keywords, video_is_on_topic,
)
import insight as insight_mod


def _cluster_views(videos) -> list[int]:
    return [v.engagement.get("views", 0) for v in videos if v.engagement.get("views")]


def _cluster_durations(videos) -> list[int]:
    return [v.duration_sec for v in videos if v.duration_sec]


def run(
    data_path: str,
    n_clusters: int,
    backend: str,
    brand_path: str | None = None,
    relevance_threshold: int | None = None,
    source: str = "json",
    limit: int = 1000,
    apify_api_query: dict | None = None,
    collect: dict | None = None,
    language: str = "ja",
    deep_video_limit: int = 0,
    deep_video_use_gemini: bool = False,
) -> list[dict]:
    """Returns the per-cluster report. If `collect` is given, it is filled with
    {"videos": list[Video], "engagement_available": bool, "deep_video":
    {"n_deepened": int, "errors": list[str]}} so the caller can run extra
    passes (e.g. sound_trends) without re-loading the data, and report how
    many clips actually got watched.

    `deep_video_limit` > 0 (needs a source that exposes a direct video URL,
    e.g. --source apify_api --platform instagram): before extracting language
    insight, the top `deep_video_limit` in-scope clips by view count are
    downloaded and actually read (spoken hook, on-screen text, format) via
    video_audit.deepen_with_video_content - same one used by the self-audit
    tab - instead of insight extraction working from caption text alone.

    `deep_video_use_gemini` picks how those clips are read: False (default)
    uses video_local - free, local, no daily-quota risk, mechanical format
    guess, no visual_tags. True sends each to Gemini's File API instead - 1
    Gemini call per clip deepened (needs a key), genuinely judged format +
    visual_tags. Either way this is on top of the 1 batched insight-
    extraction call insight_mod.extract_insights_batch always makes."""
    if brand_path is None:
        brand_path = config.BRAND_PATH
    if relevance_threshold is None:
        relevance_threshold = config.RELEVANCE_THRESHOLD

    # 1. Ingest -----------------------------------------------------------------
    if source == "huggingface":
        videos = load_videos_from_huggingface(limit=limit)
        print(f"Loaded {len(videos)} videos from Hugging Face "
              f"(lingbow/tiktok-video-engagement-200k, limit={limit})")
    elif source == "apify":
        videos = load_videos_from_apify(data_path)
        print(f"Loaded {len(videos)} de-duped videos from Apify JSON at {data_path}")
    elif source == "apify_api":
        q = apify_api_query or {}
        platform = q.get("platform", "tiktok")
        videos = load_videos_from_apify_api(
            platform=platform,
            hashtags=q.get("hashtags"),
            profiles=q.get("profiles"),
            post_urls=q.get("post_urls"),
            results_limit=q.get("results_limit", config.APIFY_RESULTS_LIMIT),
            token=config.APIFY_API_TOKEN,
            tiktok_actor=config.APIFY_TIKTOK_ACTOR,
            instagram_actor=config.APIFY_INSTAGRAM_ACTOR,
            sync_timeout=config.APIFY_SYNC_TIMEOUT,
            reels_only=q.get("reels_only", False),
        )
        print(f"Loaded {len(videos)} de-duped videos from the Apify API ({platform})")
    elif source == "json":
        videos = load_videos(data_path)
        print(f"Loaded {len(videos)} videos from {data_path}")
    else:
        raise ValueError(f"Unknown source: {source!r}")

    engagement_available = any(v.engagement.get("views") for v in videos)
    if collect is not None:
        collect["videos"] = videos
        collect["engagement_available"] = engagement_available
    print(f"Embedding backend: {backend}  |  clusters: {n_clusters}  |  "
          f"engagement: {'yes' if engagement_available else 'no'}")

    brand = load_brand(brand_path)
    print(f"Brand: {brand.get('name', '?')}  |  gate: score >= {relevance_threshold} "
          f"AND >= {config.MIN_DISTINCT_KEYWORDS} distinct brand keywords\n")

    # 2. Embed + cluster (unchanged) -----------------------------------------
    labels, _embeddings = cluster_videos(videos, n_clusters, backend)
    clusters = group_by_cluster(videos, labels)

    # 3. Relevance gate: job-hunt / career enough to deserve a language insight?
    #    Needs BOTH a high share of keyword-hitting videos AND several distinct
    #    brand keywords, so one hashtag collision (#ats = American Truck
    #    Simulator) can't drag an off-topic cluster in.
    gate = {cid: gatekeeper_score(v, brand) for cid, v in clusters.items()}
    distinct = {cid: distinct_brand_keywords(v, brand) for cid, v in clusters.items()}

    def _in_scope(cid: int) -> bool:
        return (gate[cid] >= relevance_threshold
                and len(distinct[cid]) >= config.MIN_DISTINCT_KEYWORDS)

    in_scope = {cid: v for cid, v in clusters.items() if _in_scope(cid)}

    # 3a. Drop individual off-topic outliers from an otherwise on-topic
    #     cluster (e.g. hashtag-spammed clips - an unrelated video tagged
    #     #jobsearch for reach, or one that drifted in by embedding
    #     similarity alone). The gate above only judges the cluster in
    #     aggregate; without this, such a clip can still surface as a
    #     "representative example" or feed the insight-extraction evidence.
    #     Falls back to the untouched cluster if literally every video in it
    #     fails the per-video check (shouldn't happen given the gate already
    #     requires a majority keyword-hit share, but never emit an empty
    #     "in-scope" cluster).
    for cid in list(in_scope.keys()):
        on_topic = [v for v in in_scope[cid] if video_is_on_topic(v, brand)]
        if on_topic:
            in_scope[cid] = on_topic
            clusters[cid] = on_topic

    # 3b. Optional: actually watch the top `deep_video_limit` in-scope clips
    #     (by view count) instead of guessing hook/format from caption text
    #     alone. Reuses video_audit's self-audit deepening helper - it already
    #     skips photo posts, sorts by views, and degrades a clip's failure to
    #     "keep the caption-only data" rather than sinking the run. Rewrite
    #     both in_scope AND clusters so insight extraction, the report's
    #     "examples", and median stats all see the real read.
    n_deepened, deepen_errors = 0, []
    if deep_video_limit > 0 and in_scope and (
        not deep_video_use_gemini or config.GEMINI_API_KEY
    ):
        import video_audit

        pooled = [v for vids in in_scope.values() for v in vids]
        deepened_pooled, n_deepened, deepen_errors = video_audit.deepen_with_video_content(
            pooled, limit=deep_video_limit, use_gemini=deep_video_use_gemini,
        )
        by_id = {v.id: v for v in deepened_pooled}
        for cid in list(in_scope.keys()):
            in_scope[cid] = [by_id.get(v.id, v) for v in in_scope[cid]]
            clusters[cid] = in_scope[cid]

    if collect is not None:
        collect["deep_video"] = {"n_deepened": n_deepened, "errors": deepen_errors}

    # 4. Language insight for the in-scope clusters. ONE Gemini call for all of
    #    them together; on any failure (or no key) every in-scope cluster gets
    #    the mechanical vocabulary/theme fallback instead - never an extra call.
    insights: dict[int, dict] = {}
    gemini_error: str | None = None
    if in_scope and config.GEMINI_API_KEY:
        try:
            insights = insight_mod.extract_insights_batch(
                in_scope, brand, engagement_available, language
            )
        except Exception as e:  # noqa: BLE001 - never let the API kill the run
            msg = str(e).strip().replace("\n", " ")
            if len(msg) > 300:
                msg = msg[:300] + "…"
            gemini_error = f"{type(e).__name__}: {msg}" if msg else type(e).__name__
            print(f"[gemini] insight extraction failed -> falling back: "
                  f"{gemini_error}", file=sys.stderr)

    fallback_needed = {cid: vids for cid, vids in in_scope.items() if cid not in insights}
    if fallback_needed:
        distinctive = distinctive_cluster_keywords(fallback_needed)
        for cid, vids in fallback_needed.items():
            fb = insight_mod.insight_fallback(
                vids, engagement_available, language, distinctive.get(cid)
            )
            if gemini_error:
                fb["needs_gemini_note"] += f"  (Gemini unavailable: {gemini_error})"
            insights[cid] = fb

    # 5. Assemble the report. The language insight is the star; a keyword header
    #    label, the demoted relevance score and (if present) view counts ride
    #    along.
    report: list[dict] = []
    for cluster_id, cluster_videos_list in clusters.items():
        header = explain_fallback(cluster_videos_list)
        is_target = cluster_id in in_scope
        views = _cluster_views(cluster_videos_list)
        durations = _cluster_durations(cluster_videos_list)
        n_slideshow = sum(1 for v in cluster_videos_list if v.is_slideshow)

        if is_target:
            exclude_reason = ""
        elif gate[cluster_id] < relevance_threshold:
            exclude_reason = f"関連度 {gate[cluster_id]}/100 がしきい値未満"
        else:
            exclude_reason = (
                f"関連ブランドキーワードが {len(distinct[cluster_id])} 種のみ"
                f"（{config.MIN_DISTINCT_KEYWORDS} 種以上必要）"
            )

        top = (
            sorted(cluster_videos_list,
                   key=lambda v: v.engagement.get("views", 0), reverse=True)
            if engagement_available else cluster_videos_list
        )
        report.append({
            "cluster_id": cluster_id,
            "size": len(cluster_videos_list),
            "label": header["label"],
            "why": header["why"],
            "source": (insights[cluster_id]["source"] if is_target else "n/a"),
            "relevance_score": gate[cluster_id],
            "distinct_keywords": distinct[cluster_id],
            "is_target": is_target,
            "exclude_reason": exclude_reason,
            "median_views": int(median(views)) if views else None,
            "median_duration_sec": int(median(durations)) if durations else None,
            "slideshow_pct": round(100 * n_slideshow / len(cluster_videos_list)),
            "examples": [
                {"caption": v.caption,
                 "on_screen_text": v.on_screen_text,
                 "creator": v.creator,
                 "url": v.url,
                 "views": v.engagement.get("views") or None,
                 # only populated for a clip that was actually watched
                 # (deep_video_limit) - "" means caption-only, as usual.
                 "spoken_hook": v.spoken_hook,
                 "detected_format": v.detected_format}
                for v in top[:3]
            ],
            "insight": insights.get(cluster_id),
        })
    return report


# --- Reporting ---------------------------------------------------------------

def _print_insight(ins: dict) -> None:
    print(f"   テーマ: {ins['theme']}")

    print("   フックの型:")
    if ins["hook_patterns"]:
        for j, hp in enumerate(ins["hook_patterns"], 1):
            print(f"     {j}. 型 : {hp['pattern']}")
            if hp["example"]:
                print(f"        例: \"{hp['example']}\"")
    else:
        print("     (GEMINI_API_KEY を設定すると抽出されます)")

    print("   フォーマット（頻度順）:")
    if ins["formats"]:
        for f in ins["formats"]:
            print(f"     - {f}")
    else:
        print("     (GEMINI_API_KEY を設定すると抽出されます)")

    print("   頻出ボキャブラリー:")
    if ins["vocabulary"]:
        print(f"     {', '.join(ins['vocabulary'])}")
    else:
        print("     (該当なし)")

    print("   感情トーン:")
    if ins["emotional_tone"]:
        print(f"     {', '.join(ins['emotional_tone'])}")
    else:
        print("     (GEMINI_API_KEY を設定すると抽出されます)")

    print("   自社ブランドへの応用:")
    if ins["brand_application"]:
        for j, ap in enumerate(ins["brand_application"], 1):
            tool = f"[{ap['tool']}] " if ap.get("tool") else ""
            print(f"     {j}. {tool}切り口: {ap['angle']}")
            if ap["suggested_hook"]:
                print(f"        フック例: \"{ap['suggested_hook']}\"")
    else:
        print("     (GEMINI_API_KEY を設定すると生成されます)")

    print(f"   注記: {ins['caveat']}")
    if ins.get("source") == "fallback" and ins.get("needs_gemini_note"):
        print(f"   （{ins['needs_gemini_note']}）")


def print_sound_trends(sounds: dict) -> None:
    rows = sounds.get("sounds", [])
    if not rows and sounds.get("n_with_sound", 0) == 0:
        return
    print("-" * 72)
    share = sounds.get("original_share_pct")
    tail = f"（うちオリジナル音声 {share}%）" if share is not None else ""
    print(f"サウンド: {sounds.get('n_with_sound', 0)}本にサウンドID付き{tail}")
    if not rows:
        print("  複数クリエイターに使い回されているサウンドは見つかりませんでした。")
    for i, s in enumerate(rows, 1):
        mv = f" / 中央値再生数 {s['median_views']:,}" if s["median_views"] else ""
        kind = "オリジナル" if s["is_original"] else "既存/ライセンス"
        print(f"  {i}. {s['sound_name']} [{kind}] — {s['post_count']}本 / "
              f"{s['creator_count']}人{mv}")
        if s["hashtags"]:
            print(f"     タグ: {', '.join(s['hashtags'])}")
    print(f"  注記: {sounds.get('caveat', '')}")


def print_report(report: list[dict]) -> None:
    targets = [c for c in report if c["is_target"]]
    others = [c for c in report if not c["is_target"]]
    engagement_available = any(
        ex["views"] for c in report for ex in c["examples"]
    )

    print("=" * 72)
    print("言語インサイトレポート — 就活・キャリア界隈の短尺動画で『使われている』表現")
    print("=" * 72)
    if engagement_available:
        print("※ 再生数・いいね等は取得済み。ただしハッシュタグ検索の一時点スナップ")
        print("  ショット（時系列でない）／再生数は累計・フォロワー未補正なので、")
        print("  サンプル内の相対比較の目安として読み、『いま伸びている』とは断定しない。\n")
    else:
        print("※ このデータには engagement（再生数・いいね）がありません。")
        print("  したがって各表現は『使われている』ことまでしか言えず、")
        print("  『伸びている／バズっている』とは断定できません。\n")

    if not targets:
        print("関連度しきい値以上のクラスタがありませんでした。")
        print("--relevance-threshold を下げるか、就活系の動画を増やしてください。\n")

    for i, c in enumerate(targets, 1):
        src = "Gemini" if c["source"] == "gemini" else "フォールバック(機械集計)"
        mv = f" / 中央値再生数 {c['median_views']:,}" if c["median_views"] else ""
        md = f" / 中央値尺 {c['median_duration_sec']}s" if c["median_duration_sec"] else ""
        ss = f" / スライドショー {c['slideshow_pct']}%" if c["slideshow_pct"] else ""
        print("-" * 72)
        print(f"■ インサイト {i}: {c['label']}")
        print(f"   クラスタ #{c['cluster_id']} / {c['size']}本 / 関連度 "
              f"{c['relevance_score']}/100{mv}{md}{ss} / 抽出: {src}")
        print(f"   一致ブランドキーワード: {', '.join(c['distinct_keywords'])}")
        print("   代表例" + ("（再生数上位）:" if engagement_available else ":"))
        for ex in c["examples"]:
            # .get(): examples from an OLDER saved reference snapshot won't
            # have spoken_hook/detected_format at all (added later).
            spoken_hook = ex.get("spoken_hook", "")
            detected_format = ex.get("detected_format", "")
            quote = spoken_hook or ex["on_screen_text"] or ex["caption"]
            quote = (quote[:140] + "…") if len(quote) > 141 else quote
            views = f" [{ex['views']:,} views]" if ex["views"] else ""
            watched = " [動画を確認済み]" if spoken_hook or detected_format else ""
            fmt = f" / format: {detected_format}" if detected_format else ""
            print(f"     - \"{quote}\"  (@{ex['creator']}){views}{watched}{fmt}")
        _print_insight(c["insight"])
        print()

    if others:
        print("-" * 72)
        print("対象外クラスタ（言語インサイトは生成せず）:")
        for c in others:
            mv = f", 中央値再生数 {c['median_views']:,}" if c["median_views"] else ""
            print(f"   - クラスタ #{c['cluster_id']} ({c['size']}本{mv}): "
                  f"{c['label']}  — {c['exclude_reason']}")
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Social Trend Agent - 就活SNSの言語インサイト抽出パイプライン"
    )
    p.add_argument("--source", default="json",
                   choices=["json", "huggingface", "apify", "apify_api"],
                   help="Where to load videos from (default: json). apify_api "
                        "runs the Apify scraper live (needs APIFY_API_TOKEN).")
    p.add_argument("--data", default=config.DATA_PATH,
                   help="Local path: JSON file (--source json) or Apify JSON "
                        "file/dir (--source apify, default: data/apify)")
    p.add_argument("--language", default="ja", choices=["ja", "en"],
                   help="Output language for themes / caveats / insights (default: ja)")
    p.add_argument("--platform", default="tiktok", choices=["tiktok", "instagram"],
                   help="--source apify_api: which scraper (default: tiktok)")
    p.add_argument("--reels-only", action="store_true", dest="reels_only",
                   help="--source apify_api --platform instagram: fetch "
                        "resultsType=reels instead of posts, so every row is "
                        "a video (no photo/slideshow posts) - maximises how "
                        "many clips --deep-video-limit can actually watch")
    p.add_argument("--hashtags", default="",
                   help="--source apify_api: comma-separated hashtags to scrape")
    p.add_argument("--profiles", default="",
                   help="--source apify_api: comma-separated handles / profile URLs")
    p.add_argument("--post-urls", default="", dest="post_urls",
                   help="--source apify_api: comma-separated post URLs")
    p.add_argument("--results-limit", type=int, default=config.APIFY_RESULTS_LIMIT,
                   dest="results_limit",
                   help="--source apify_api: max rows PER hashtag/profile/URL, "
                        "not a total - multiple ones multiply the total fetched "
                        f"(default: {config.APIFY_RESULTS_LIMIT})")
    p.add_argument("--limit", type=int, default=1000,
                   help="Max rows to pull (--source huggingface only, default: 1000)")
    p.add_argument("--clusters", type=int, default=config.N_CLUSTERS)
    p.add_argument("--backend", default=config.EMBEDDING_BACKEND,
                   choices=["sentence-transformers", "tfidf"])
    p.add_argument("--brand", default=config.BRAND_PATH,
                   help="Path to a brand profile JSON (default: data/brand.json)")
    p.add_argument("--relevance-threshold", type=int, default=config.RELEVANCE_THRESHOLD,
                   help="Clusters scoring >= this (0-100) get a language insight")
    p.add_argument("--out", default=".",
                   help="Directory to write report.json + report.md (default: .)")
    p.add_argument("--no-save", action="store_true",
                   help="Print the report only; don't write report.json / report.md")
    p.add_argument("--no-reference", action="store_true",
                   help="Don't save a self-audit reference snapshot "
                        f"(into {config.REFERENCE_HISTORY_DIR}, keeps newest "
                        f"{config.REFERENCE_HISTORY_KEEP})")
    p.add_argument("--no-gemini", action="store_true",
                   help="Force offline: don't call Gemini even if a key is set")
    p.add_argument("--deep-video-limit", type=int, default=0, dest="deep_video_limit",
                   help="Actually watch up to this many in-scope clips by "
                        "view count, instead of inferring hook/format from "
                        "caption text alone. Needs a source with a direct "
                        "video URL (e.g. --source apify_api --platform "
                        "instagram). Default: 0 (off)")
    p.add_argument("--deep-video-gemini", action="store_true",
                   dest="deep_video_use_gemini",
                   help="Use Gemini's File API (1 call per clip, needs a "
                        "key) for --deep-video-limit instead of the default "
                        "free local reader (faster-whisper + EasyOCR + "
                        "PySceneDetect, no quota risk but a mechanical "
                        "format guess and no visual_tags).")
    args = p.parse_args()

    if args.no_gemini:
        config.GEMINI_API_KEY = ""

    data_path = args.data
    if args.source == "apify" and args.data == config.DATA_PATH:
        data_path = config.APIFY_PATH

    def _split(s: str) -> list[str]:
        return [x.strip() for x in s.split(",") if x.strip()]

    apify_api_query = None
    if args.source == "apify_api":
        apify_api_query = {
            "platform": args.platform,
            "hashtags": _split(args.hashtags),
            "profiles": _split(args.profiles),
            "post_urls": _split(args.post_urls),
            "results_limit": args.results_limit,
            "reels_only": args.reels_only,
        }
        if not any(apify_api_query[k] for k in ("hashtags", "profiles", "post_urls")):
            p.error("--source apify_api needs one of --hashtags / --profiles / --post-urls")

    collect: dict = {}
    report = run(data_path, args.clusters, args.backend, args.brand,
                 args.relevance_threshold, args.source, args.limit,
                 apify_api_query=apify_api_query, collect=collect,
                 language=args.language, deep_video_limit=args.deep_video_limit,
                 deep_video_use_gemini=args.deep_video_use_gemini)
    print_report(report)

    dv = collect.get("deep_video", {})
    if dv.get("n_deepened"):
        print(f"\n[deep video] actually watched {dv['n_deepened']} clip(s).")
    if dv.get("errors"):
        print(f"[deep video] {len(dv['errors'])} clip(s) failed and stayed "
              f"caption-only: {'; '.join(dv['errors'])}", file=sys.stderr)

    import sound_trends
    sounds = sound_trends.analyse_sounds(
        collect.get("videos", []), collect.get("engagement_available", False),
        language=args.language,
    )
    print_sound_trends(sounds)

    if not args.no_save:
        json_path, md_path = exporter.write(report, args.out)
        print(f"\nSaved: {json_path}  |  {md_path}")

    if not args.no_reference:
        ref_path = exporter.save_reference(
            report, config.REFERENCE_HISTORY_DIR,
            meta={
                "source": args.source,
                "n_clusters": args.clusters,
                "relevance_threshold": args.relevance_threshold,
                "gemini": bool(config.GEMINI_API_KEY),
                "language": args.language,
            },
            sounds=sounds,
            keep=config.REFERENCE_HISTORY_KEEP,
        )
        print(f"Reference cache: {ref_path}")


if __name__ == "__main__":
    main()
