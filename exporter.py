"""
Output layer: turn a pipeline report into shareable files.

    report.json  - the exact structured list run() returns, for downstream
                   agents / diffing runs. UTF-8, pretty-printed.
    report.md    - a human-readable rendering, same content as print_report,
                   for a marketer to paste into a doc.

Both always carry the caveat (the honest "hashtag-search snapshot / cumulative
counts / not "trending"" note) so it can't be lost once the report leaves the
terminal. Notion / Slack push can be layered on top later; keep them opt-in and
make them reuse report_markdown() so the caveat still travels.
"""
from __future__ import annotations
import datetime
import json
import os


def _engagement_available(report: list[dict]) -> bool:
    return any(ex["views"] for c in report for ex in c["examples"])


def report_json(report: list[dict]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)


def _insight_md(ins: dict) -> list[str]:
    L = [f"**テーマ**: {ins['theme']}", ""]

    L.append("**フックの型**")
    if ins["hook_patterns"]:
        for hp in ins["hook_patterns"]:
            ex = f" — 例: 「{hp['example']}」" if hp["example"] else ""
            L.append(f"- `{hp['pattern']}`{ex}")
    else:
        L.append("- _(GEMINI_API_KEY を設定すると抽出されます)_")
    L.append("")

    L.append("**フォーマット（頻度順）**: "
             + (", ".join(ins["formats"]) if ins["formats"]
                else "_(GEMINI_API_KEY を設定すると抽出されます)_"))
    L.append("")
    L.append("**頻出ボキャブラリー**: "
             + (", ".join(ins["vocabulary"]) if ins["vocabulary"] else "_(該当なし)_"))
    L.append("")
    L.append("**感情トーン**: "
             + (", ".join(ins["emotional_tone"]) if ins["emotional_tone"]
                else "_(GEMINI_API_KEY を設定すると抽出されます)_"))
    L.append("")

    L.append("**自社ブランドへの応用**")
    if ins["brand_application"]:
        for ap in ins["brand_application"]:
            tool = f"[{ap['tool']}] " if ap.get("tool") else ""
            L.append(f"- {tool}**切り口**: {ap['angle']}")
            if ap["suggested_hook"]:
                L.append(f"  - フック例: 「{ap['suggested_hook']}」")
    else:
        L.append("- _(GEMINI_API_KEY を設定すると生成されます)_")
    L.append("")

    L.append(f"> 注記: {ins['caveat']}")
    if ins.get("source") == "fallback" and ins.get("needs_gemini_note"):
        L.append(f">")
        L.append(f"> （{ins['needs_gemini_note']}）")
    return L


def report_markdown(report: list[dict]) -> str:
    targets = [c for c in report if c["is_target"]]
    others = [c for c in report if not c["is_target"]]
    engagement = _engagement_available(report)

    L = ["# 言語インサイトレポート — 就活・キャリア界隈の短尺動画で『使われている』表現", ""]
    if engagement:
        L += ["> ※ 再生数・いいね等は取得済み。ただしハッシュタグ検索の一時点スナップショット"
              "（時系列でない）・累計値・フォロワー未補正。サンプル内の相対比較の目安として"
              "読み、『いま伸びている』とは断定しないこと。", ""]
    else:
        L += ["> ※ このデータには engagement（再生数・いいね）がありません。各表現は"
              "『使われている』ことまでしか言えず、『伸びている／バズっている』とは"
              "断定できません。", ""]

    if not targets:
        L += ["_関連度しきい値以上のクラスタがありませんでした。_", ""]

    for i, c in enumerate(targets, 1):
        src = "Gemini" if c["source"] == "gemini" else "フォールバック(機械集計)"
        meta = [f"クラスタ #{c['cluster_id']}", f"{c['size']}本",
                f"関連度 {c['relevance_score']}/100"]
        if c["median_views"]:
            meta.append(f"中央値再生数 {c['median_views']:,}")
        if c["median_duration_sec"]:
            meta.append(f"中央値尺 {c['median_duration_sec']}s")
        if c["slideshow_pct"]:
            meta.append(f"スライドショー {c['slideshow_pct']}%")
        meta.append(f"抽出: {src}")

        L += [f"## {i}. {c['label']}", "", " · ".join(meta), ""]
        L.append(f"一致ブランドキーワード: {', '.join(c['distinct_keywords'])}")
        L.append("")
        L.append("**代表例" + ("（再生数上位）**" if engagement else "**"))
        for ex in c["examples"]:
            # .get(): examples from an OLDER saved reference snapshot won't
            # have spoken_hook/detected_format at all (added later).
            spoken_hook = ex.get("spoken_hook", "")
            detected_format = ex.get("detected_format", "")
            quote = spoken_hook or ex["on_screen_text"] or ex["caption"]
            quote = (quote[:140] + "…") if len(quote) > 141 else quote
            v = f" · {ex['views']:,} views" if ex["views"] else ""
            link = f" ([link]({ex['url']}))" if ex["url"] else ""
            watched = " [動画確認済み]" if spoken_hook or detected_format else ""
            fmt = f" · format: {detected_format}" if detected_format else ""
            L.append(f"- 「{quote}」 (@{ex['creator']}){v}{watched}{fmt}{link}")
        L.append("")
        L += _insight_md(c["insight"])
        L.append("")

    if others:
        L += ["## 対象外クラスタ（言語インサイトは生成せず）", ""]
        for c in others:
            mv = f", 中央値再生数 {c['median_views']:,}" if c["median_views"] else ""
            L.append(f"- クラスタ #{c['cluster_id']} ({c['size']}本{mv}): "
                     f"{c['label']} — {c['exclude_reason']}")
        L.append("")

    return "\n".join(L)


def write(report: list[dict], out_dir: str = ".") -> tuple[str, str]:
    """Write report.json + report.md into out_dir. Returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "report.json")
    md_path = os.path.join(out_dir, "report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(report_json(report))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_markdown(report))
    return json_path, md_path


# --- Reference history (so the self-audit tab need not re-run, and old runs
# aren't silently lost) -------------------------------------------------------
# Every niche run writes a NEW timestamped snapshot into a history directory
# instead of overwriting a single file, and only the newest `keep` are retained.
# The self-audit tab lists them and lets the user pick which one to score
# against, rather than always getting whatever ran last.

_LEGACY_SINGLE_FILE = "data/reference_report.json"


def save_reference(
    report: list[dict], history_dir: str, meta: dict | None = None,
    sounds: dict | None = None, keep: int = 3,
) -> str:
    """Write a new timestamped snapshot into `history_dir`, then prune to the
    `keep` newest. Returns the new snapshot's path."""
    os.makedirs(history_dir, exist_ok=True)
    now = datetime.datetime.now().astimezone()
    path = os.path.join(history_dir, now.strftime("%Y%m%d_%H%M%S_%f") + ".json")
    payload = {
        "saved_at": now.isoformat(timespec="seconds"),
        "meta": meta or {},
        "report": report,
        "sounds": sounds or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    snapshots = sorted(
        (f for f in os.listdir(history_dir) if f.endswith(".json")), reverse=True,
    )
    for stale in snapshots[keep:]:
        try:
            os.remove(os.path.join(history_dir, stale))
        except OSError:
            pass
    return path


def _migrate_legacy_reference(history_dir: str) -> None:
    """One-time upgrade: if an old single-file reference.json exists and the
    new history directory has nothing yet, move it in as the first snapshot
    instead of silently discarding a perfectly good saved reference."""
    if os.path.isdir(history_dir) and any(
        f.endswith(".json") for f in os.listdir(history_dir)
    ):
        return
    if not os.path.isfile(_LEGACY_SINGLE_FILE):
        return
    report, info = load_reference(_LEGACY_SINGLE_FILE)
    if report is None:
        return
    save_reference(report, history_dir, meta=(info or {}).get("meta"),
                    sounds=(info or {}).get("sounds"))
    try:
        os.remove(_LEGACY_SINGLE_FILE)
    except OSError:
        pass


def list_references(history_dir: str) -> list[dict]:
    """[{path, saved_at, meta}] for every saved snapshot, newest first.
    Migrates the old single-file reference.json in on first call if present.
    """
    _migrate_legacy_reference(history_dir)
    if not os.path.isdir(history_dir):
        return []
    out = []
    for name in sorted(os.listdir(history_dir), reverse=True):
        if not name.endswith(".json"):
            continue
        path = os.path.join(history_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            continue
        out.append({
            "path": path,
            "saved_at": payload.get("saved_at", ""),
            "meta": payload.get("meta", {}),
        })
    return out


def load_reference(path: str) -> tuple[list[dict] | None, dict | None]:
    """Return (report, info) from one saved snapshot file, or (None, None) if
    it is missing, unreadable, or malformed. `info` = {saved_at, meta, sounds}."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None, None
    report = payload.get("report")
    if not isinstance(report, list) or not report:
        return None, None
    return report, {
        "saved_at": payload.get("saved_at", ""),
        "meta": payload.get("meta", {}),
        "sounds": payload.get("sounds", {}),
    }


# --- Self-audit run history --------------------------------------------------
# Same "never overwritten, newest N kept" shape as the reference history above,
# so a past self-audit (own clips scored against a niche reference) can be
# reopened without re-spending a Gemini + Apify call on it. The audit tab
# lists these and lets the user pick one to view, alongside "run a new audit".

def save_audit(
    items: list[dict], videos: dict, summary: dict | None, history_dir: str,
    meta: dict | None = None, keep: int = 10,
) -> str:
    """Write a new timestamped self-audit snapshot into `history_dir`, then
    prune to the `keep` newest. Returns the new snapshot's path."""
    os.makedirs(history_dir, exist_ok=True)
    now = datetime.datetime.now().astimezone()
    path = os.path.join(history_dir, now.strftime("%Y%m%d_%H%M%S_%f") + ".json")
    payload = {
        "saved_at": now.isoformat(timespec="seconds"),
        "meta": meta or {},
        "items": items,
        "videos": {vid: v.as_dict() for vid, v in videos.items()},
        "summary": summary or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    snapshots = sorted(
        (f for f in os.listdir(history_dir) if f.endswith(".json")), reverse=True,
    )
    for stale in snapshots[keep:]:
        try:
            os.remove(os.path.join(history_dir, stale))
        except OSError:
            pass
    return path


def list_audits(history_dir: str) -> list[dict]:
    """[{path, saved_at, meta}] for every saved self-audit snapshot, newest
    first."""
    if not os.path.isdir(history_dir):
        return []
    out = []
    for name in sorted(os.listdir(history_dir), reverse=True):
        if not name.endswith(".json"):
            continue
        path = os.path.join(history_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            continue
        out.append({
            "path": path,
            "saved_at": payload.get("saved_at", ""),
            "meta": payload.get("meta", {}),
        })
    return out


def load_audit(path: str):
    """Return (items, videos, summary, info) from one saved self-audit
    snapshot, or (None, None, None, None) if missing/unreadable/malformed.
    `videos` is rebuilt as {video_id: ingest.Video}, same shape the audit tab
    keeps in st.session_state, so a loaded run renders exactly like a fresh
    one. `info` = {saved_at, meta}."""
    from ingest import Video

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None, None, None, None
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return None, None, None, None
    videos = {}
    for vid, d in (payload.get("videos") or {}).items():
        if not isinstance(d, dict):
            continue
        known = {k: d[k] for k in Video.__dataclass_fields__ if k in d}
        videos[vid] = Video(**known)
    return items, videos, payload.get("summary", {}), {
        "saved_at": payload.get("saved_at", ""),
        "meta": payload.get("meta", {}),
    }


# --- Self-audit export (download from the "② Audit your posts" tab) --------

_AUDIT_LABELS = {
    "ja": {
        "title": "自己診断レポート",
        "generated": "生成日時",
        "scoreboard": "スコアボード",
        "clips": "診断本数",
        "high": "優先度・高",
        "overall": "全体所見",
        "recs": "横断的な改善アクション",
        "vocab_gaps": "共通して不足している語彙",
        "hook": "検出したフック",
        "format": "検出したフォーマット",
        "url": "投稿URL",
        "engagement": "エンゲージメント",
        "no_views_photo": "写真投稿のため再生数なし",
        "assessment": "総評",
        "works": "できている点",
        "gaps": "ギャップ",
        "rewrites": "フック書き換え案",
        "fmt_suggestion": "フォーマット提案",
        "vocab_adopt": "取り入れたい語彙",
        "priority": "優先度",
        "caveat": "注記",
    },
    "en": {
        "title": "Self-Audit Report",
        "generated": "Generated",
        "scoreboard": "Scoreboard",
        "clips": "Clips audited",
        "high": "High priority",
        "overall": "Overall read",
        "recs": "Cross-cutting actions",
        "vocab_gaps": "Vocabulary missing across clips",
        "hook": "Detected hook",
        "format": "Detected format",
        "url": "Post URL",
        "engagement": "Engagement",
        "no_views_photo": "Photo post — no view count",
        "assessment": "Assessment",
        "works": "What works",
        "gaps": "Gaps",
        "rewrites": "Hook rewrites",
        "fmt_suggestion": "Format suggestion",
        "vocab_adopt": "Vocabulary to adopt",
        "priority": "Priority",
        "caveat": "Caveat",
    },
}


def audit_json(items: list[dict], videos: dict, summary: dict | None) -> str:
    """JSON export of a self-audit run: the scoreboard + every per-clip result,
    each clip merged with what was detected from the video itself."""
    payload = {
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": summary or {},
        "items": [
            {**item, "video": videos[item["video_id"]].as_dict()}
            if item["video_id"] in videos else item
            for item in items
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _audit_engagement_line(v, no_views_photo_label: str = "") -> str:
    """views/likes/... formatted for the audit export, or '' if none - so a
    reader can independently check the numbers against the live post. A photo
    /carousel post has no view count on Instagram (not a data bug) - flagged
    with `no_views_photo_label` instead of silently omitting it."""
    parts = [f"{k}: {n:,}" for k, n in v.engagement.items() if n]
    if not v.engagement.get("views") and v.is_slideshow and v.engagement_is_real:
        parts.insert(0, no_views_photo_label or "no view count (photo post)")
    return ", ".join(parts)


def audit_markdown(
    items: list[dict], videos: dict, summary: dict | None, language: str = "ja",
) -> str:
    """Human-readable Markdown export of a self-audit run."""
    L_ = _AUDIT_LABELS.get(language, _AUDIT_LABELS["ja"])
    now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    lines = [f"# {L_['title']}", "", f"{L_['generated']}: {now}", ""]

    if summary:
        pc = summary.get("priority_counts", {})
        lines += [f"## {L_['scoreboard']}", ""]
        lines.append(f"- {L_['clips']}: {summary.get('n_clips', 0)}")
        lines.append(f"- {L_['high']}: {pc.get('high', 0)}")
        if summary.get("overall"):
            lines += ["", f"**{L_['overall']}**", "", summary["overall"]]
        if summary.get("recommendations"):
            lines += ["", f"**{L_['recs']}**", ""]
            lines += [f"- {r}" for r in summary["recommendations"]]
        gaps = summary.get("top_vocabulary_gaps", [])
        if gaps:
            lines += ["", f"**{L_['vocab_gaps']}**: "
                      + ", ".join(f"{g['term']} ({g['count']})" for g in gaps)]
        lines.append("")

    for item in items:
        v = videos.get(item["video_id"])
        lines += [f"## {item['video_id']}  ({L_['priority']}: {item['priority']})", ""]
        if v is not None:
            if v.spoken_hook:
                lines.append(f"- {L_['hook']}: 「{v.spoken_hook}」")
            if v.detected_format:
                lines.append(f"- {L_['format']}: {v.detected_format}")
            if v.url:
                # Angle brackets are CommonMark's explicit autolink form, so
                # this renders as a clickable link in every Markdown viewer
                # (a bare URL is only auto-linked by some renderers).
                lines.append(f"- {L_['url']}: <{v.url}>")
            eng = _audit_engagement_line(v, L_["no_views_photo"])
            if eng:
                lines.append(f"- {L_['engagement']}: {eng}")
        if item.get("assessment"):
            lines += ["", f"**{L_['assessment']}**", "", item["assessment"]]
        if item.get("what_works"):
            lines += ["", f"**{L_['works']}**", ""]
            lines += [f"- {x}" for x in item["what_works"]]
        if item.get("gaps"):
            lines += ["", f"**{L_['gaps']}**", ""]
            lines += [f"- {x}" for x in item["gaps"]]
        if item.get("hook_rewrites"):
            lines += ["", f"**{L_['rewrites']}**", ""]
            for rw in item["hook_rewrites"]:
                lines.append(
                    f"- {rw['original'] or '—'} → **{rw['rewrite']}** "
                    f"(based on: {rw['based_on'] or '—'})"
                )
        if item.get("format_suggestion"):
            lines += ["", f"**{L_['fmt_suggestion']}**: {item['format_suggestion']}"]
        if item.get("vocabulary_to_adopt"):
            lines += ["", f"**{L_['vocab_adopt']}**: " + ", ".join(item["vocabulary_to_adopt"])]
        lines += ["", f"> {L_['caveat']}: {item.get('caveat', '')}", ""]

    return "\n".join(lines)


def audit_pdf(
    items: list[dict], videos: dict, summary: dict | None, language: str = "ja",
) -> bytes:
    """PDF export of a self-audit run, for printing / sharing outside the app.

    Uses reportlab's built-in CID font (HeiseiKakuGo-W5) rather than a bundled
    TTF, so Japanese renders correctly with no font file to ship - it covers
    the Latin/English content too, so one font serves both UI languages.
    """
    import io
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer,
    )

    font = "HeiseiKakuGo-W5"
    if font not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(font))

    L_ = _AUDIT_LABELS.get(language, _AUDIT_LABELS["ja"])
    styles = {
        "h1": ParagraphStyle("h1", fontName=font, fontSize=17, leading=22,
                              spaceAfter=4, textColor=colors.HexColor("#262420")),
        "h2": ParagraphStyle("h2", fontName=font, fontSize=13, leading=18,
                              spaceBefore=10, spaceAfter=6,
                              textColor=colors.HexColor("#262420")),
        "label": ParagraphStyle("label", fontName=font, fontSize=10, leading=14,
                                 spaceBefore=6, textColor=colors.HexColor("#5F6B57")),
        "body": ParagraphStyle("body", fontName=font, fontSize=10, leading=15,
                                textColor=colors.HexColor("#262420")),
        "bullet": ParagraphStyle("bullet", fontName=font, fontSize=10, leading=15,
                                  leftIndent=12,
                                  textColor=colors.HexColor("#262420")),
        "caveat": ParagraphStyle("caveat", fontName=font, fontSize=9, leading=13,
                                  spaceBefore=8, textColor=colors.HexColor("#8C877C")),
        "meta": ParagraphStyle("meta", fontName=font, fontSize=9, leading=13,
                                textColor=colors.HexColor("#8C877C")),
    }

    def P(text: str, style: str = "body"):
        return Paragraph(escape(str(text)), styles[style])

    def P_link(label: str, url: str, style: str = "meta"):
        """A label followed by a clickable hyperlink (reportlab's Paragraph
        mini-markup supports <a href>; P() above escapes that away, so a
        plain P() call would print the URL as text but not link it)."""
        safe_url = escape(url, {'"': "&quot;"})
        markup = (
            f'{escape(label)}: <a href="{safe_url}" color="#3F6634">'
            f'{escape(url)}</a>'
        )
        return Paragraph(markup, styles[style])

    story = []
    now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    story.append(P(L_["title"], "h1"))
    story.append(P(f"{L_['generated']}: {now}", "meta"))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#E4DED2")))

    if summary:
        pc = summary.get("priority_counts", {})
        story.append(P(L_["scoreboard"], "h2"))
        story.append(P(f"{L_['clips']}: {summary.get('n_clips', 0)}"
                        f"　·　{L_['high']}: {pc.get('high', 0)}", "body"))
        if summary.get("overall"):
            story.append(P(L_["overall"], "label"))
            story.append(P(summary["overall"], "body"))
        if summary.get("recommendations"):
            story.append(P(L_["recs"], "label"))
            for r in summary["recommendations"]:
                story.append(P(f"•  {r}", "bullet"))
        gaps = summary.get("top_vocabulary_gaps", [])
        if gaps:
            story.append(P(L_["vocab_gaps"], "label"))
            story.append(P(", ".join(f"{g['term']} ({g['count']})" for g in gaps), "body"))

    for item in items:
        v = videos.get(item["video_id"])
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", color=colors.HexColor("#E4DED2")))
        story.append(P(f"{item['video_id']}　·　{L_['priority']}: {item['priority']}", "h2"))
        if v is not None:
            if v.spoken_hook:
                story.append(P(f"{L_['hook']}: 「{v.spoken_hook}」", "body"))
            if v.detected_format:
                story.append(P(f"{L_['format']}: {v.detected_format}", "body"))
            if v.url:
                story.append(P_link(L_["url"], v.url, "meta"))
            eng = _audit_engagement_line(v, L_["no_views_photo"])
            if eng:
                story.append(P(f"{L_['engagement']}: {eng}", "meta"))
        if item.get("assessment"):
            story.append(P(L_["assessment"], "label"))
            story.append(P(item["assessment"], "body"))
        if item.get("what_works"):
            story.append(P(L_["works"], "label"))
            for x in item["what_works"]:
                story.append(P(f"•  {x}", "bullet"))
        if item.get("gaps"):
            story.append(P(L_["gaps"], "label"))
            for x in item["gaps"]:
                story.append(P(f"•  {x}", "bullet"))
        if item.get("hook_rewrites"):
            story.append(P(L_["rewrites"], "label"))
            for rw in item["hook_rewrites"]:
                story.append(P(
                    f"{rw['original'] or '—'} → {rw['rewrite']} "
                    f"(based on: {rw['based_on'] or '—'})", "bullet",
                ))
        if item.get("format_suggestion"):
            story.append(P(f"{L_['fmt_suggestion']}: {item['format_suggestion']}", "body"))
        if item.get("vocabulary_to_adopt"):
            story.append(P(
                f"{L_['vocab_adopt']}: " + ", ".join(item["vocabulary_to_adopt"]), "body",
            ))
        story.append(P(f"{L_['caveat']}: {item.get('caveat', '')}", "caveat"))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=20 * mm, bottomMargin=20 * mm,
        leftMargin=20 * mm, rightMargin=20 * mm,
    )
    doc.build(story)
    return buf.getvalue()
