"""
Streamlit UI for the language-insight pipeline.

Mirrors pipeline.print_report: for each job-hunt / career cluster that clears
the relevance gate, show theme -> hook patterns -> formats -> vocabulary ->
emotional tone -> brand application -> caveat.

The caveat matches the data source: the sample / Hugging Face sets have no
engagement (only "being used"); the Apify scrape has view counts, used for
relative comparison within the sample only (never "trending right now").

UI chrome is bilingual (日本語 / English). Insight CONTENT (theme, hooks,
vocabulary, ...) comes from the pipeline itself - Gemini is prompted to answer
in Japanese, the mechanical fallback echoes the source language - so it is not
re-translated here.

Run:
    streamlit run app.py
"""
from __future__ import annotations
import os

import streamlit as st

import config
import exporter
from pipeline import run

# --- i18n: UI chrome only ---------------------------------------------------
T = {
    "ja": {
        "page_title": "Social Trend Agent - 言語インサイト",
        "hero_eyebrow": "SOCIAL TREND AGENT · SOCIAL LANGUAGE INTELLIGENCE",
        "title": "就活SNS 言語インサイト",
        "subtitle": (
            "就活・キャリア界隈の短尺動画で「いま使われている」表現を調べ、"
            "自社の投稿をそれと比べて改善案を出すツールです。"
        ),
        "how_title": "はじめての方へ（使い方）",
        "how_body": (
            "**2ステップで使います。**\n\n"
            "1. **「① 業界を調べる」タブ** … 就活・キャリア界隈の短尺動画をまとめて解析し、"
            "テーマごとに「よく使われるフック（最初の一言）・動画の形式・言い回し・感情トーン」を"
            "抽出します。まずは設定を変えずに **「分析を実行」** を押してください。\n"
            "2. **「② 自分の投稿を診断」タブ** … 自社ブランドの投稿動画を入れると、"
            "①で分かった業界パターンと照らし合わせて、フックの弱い点や取り入れるべき言い回しを"
            "具体的に指摘します。\n\n"
            "※ このツールは「使われている」表現までしか言いません。「これをやれば伸びる／バズる」"
            "とは断定しません。"
        ),
        "glossary_title": "用語ガイド",
        "glossary_body": (
            "- **フック** … 動画の最初の2〜3秒の一言。視聴継続を決める部分。\n"
            "- **フックの型** … 「POV: {状況}」のような、穴埋めで使えるパターン。\n"
            "- **フォーマット** … 動画の作り方（顔出し＋テロップ／スライドショー／グリーンバック等）。\n"
            "- **クラスタ／テーマ** … 似た動画を機械的にまとめたグループ。1グループ＝1つの話題。\n"
            "- **関連度** … そのグループが就活・キャリアの話にどれだけ寄っているか（0〜100）。\n"
            "- **インサイト** … 各テーマについて抽出した、投稿制作にそのまま使える要点。"
        ),
        "advanced": "詳細設定（通常は変更不要）",
        "no_engagement": (
            "このデータには engagement（再生数・いいね）がありません。"
            "各表現は『使われている』ことまでしか言えず、"
            "『伸びている／バズっている』とは断定できません。"
        ),
        "with_engagement": (
            "再生数・いいね等は取得済み。ただしハッシュタグ検索の一時点スナップ"
            "ショット（時系列でない）・累計値・フォロワー未補正なので、"
            "サンプル内の相対比較の目安として読み、『いま伸びている』とは断定しない。"
        ),
        "language": "言語 / Language",
        "settings": "設定",
        "data_source": "調べるデータ",
        "source_json": "お試し用サンプル（すぐ動く）",
        "source_hf": "公開データセット（実データ・大量）",
        "source_apify": "取得済みApifyファイル（実データ）",
        "source_apify_api": "Apifyでいま取得する（実データ）",
        "apify_path": "Apify JSON のパス（ファイル or ディレクトリ）",
        "apify_platform": "プラットフォーム",
        "apify_hashtags": "ハッシュタグ（カンマ区切り）",
        "apify_reels_only": "Reels（動画）のみ取得する",
        "apify_reels_only_help": (
            "OFFだと写真投稿（スライドショー）も混ざり、それらには動画が無いため"
            "「動画の中身も見る」機能で解析できません。ONにすると投稿を動画"
            "（Reels）に絞って取得するので、取得した投稿がすべて解析対象になります。"
        ),
        "apify_results_limit_hashtag": "ハッシュタグごとの動画取得上限",
        "apify_results_limit_account": "アカウントごとの動画取得上限",
        "apify_results_limit_help": (
            "合計ではなく、ハッシュタグ・プロフィール・投稿URLの1件ごとの上限です。"
            "複数指定すると取得件数・処理時間・Apifyの消費もその分増えます"
            "（例: ハッシュタグ2つ×上限100 → 最大200件、重複除去前）。"
        ),
        "apify_no_token": "APIFY_API_TOKEN が .env に未設定です。取得できません。",
        "niche_deep": "🎬 再生数上位の動画も実際に見る",
        "niche_deep_help": (
            "クラスタの対象になった動画のうち、再生数が多い順に指定本数まで実際に"
            "ダウンロードして中身を読み、フック・フォーマット・画面内テキストを"
            "本当に観察した上でインサイトを抽出します。デフォルトは無料のローカル"
            "解析（faster-whisper＋EasyOCR＋PySceneDetect）でAPIは消費しません。"
        ),
        "niche_deep_limit": "解析する本数の上限",
        "niche_deep_limit_help": "残りの動画はキャプションのみからの推測になります。",
        "niche_deep_tiktok_note": (
            "※ 動画の中身も見る機能は今のところInstagram取得のみ対応です"
            "（TikTokスクレイパーは動画への直リンクを返さないため）。"
        ),
        "apify_need_query": "ハッシュタグを1つ以上入力してください。",
        "data_file": "データファイル",
        "median_views": "中央値再生数",
        "matched_keywords": "一致ブランドキーワード",
        "download_json": "report.json をダウンロード",
        "download_md": "report.md をダウンロード",
        "download_audit_json": "診断結果を JSON でダウンロード",
        "download_audit_md": "診断結果を Markdown でダウンロード",
        "download_audit_pdf": "診断結果を PDF でダウンロード",
        "hf_limit": "取得件数（Hugging Face）",
        "backend": "文章の解析方法",
        "backend_help": "tfidf はネット不要で高速。sentence-transformers は精度が高いが初回にモデルをダウンロード。",
        "n_clusters": "見つけるテーマの数",
        "brand_profile": "ブランド設定ファイルのパス",
        "relevance_threshold": "就活・キャリアにどれだけ絞るか（高いほど厳しく絞る）",
        "use_gemini": "AIで詳しく分析する（推奨）",
        "use_gemini_help": (
            "オンにすると Gemini がフック・フォーマット・改善案まで生成します。"
            "オフだと言葉の頻度集計だけの簡易結果になります（キーがあっても呼びません）。"
        ),
        "warn_no_key": "AIキー（GEMINI_API_KEY）が未設定です。簡易集計のみで動作します。",
        "info_offline": "簡易モード：AIは使わず、言葉の頻度集計だけを表示します。",
        "run": "分析を実行",
        "spinner": "分析中…",
        "error_prefix": "実行中にエラーが発生しました",
        "prompt_run": (
            "**ステップ①です。** 左の設定はそのままで「分析を実行」を押すと、"
            "就活SNSで使われている表現パターンを抽出します。"
        ),
        "metric_clusters": "見つけたテーマ数",
        "metric_videos": "分析した動画数",
        "metric_targets": "就活・キャリア関連のテーマ",
        "no_targets": "就活・キャリアに十分寄ったテーマが見つかりませんでした。詳細設定で「就活にどれだけ絞るか」を下げて再実行してください。",
        "src_gemini": "🟢 AI分析",
        "src_fallback": "⚪️ 簡易集計",
        "cluster_meta": "動画 {n}本 · 就活関連度 {score}/100 · キーワード: {label}",
        "theme_card_meta": "{n}本 · 関連度{score}",
        "theme_grid_hint": "気になるテーマをクリックすると詳細が下に表示されます。",
        "examples": "代表例",
        "examples_top": "代表例（再生数上位）",
        "hook_patterns": "フックの型",
        "hook_patterns_video": "フックの型（動画確認済み）",
        "hook_patterns_caption": "キャプションパターン（動画未確認）",
        "hook_patterns_caption_note": (
            "※ 動画本体は見ておらず、キャプション文面から推測した言い回しです。"
            "実際の動画内で使われるフックとはズレることがあります。"
            "「再生数上位の動画も実際に見る」をONにすると、実際に見た動画から"
            "検出したフックに切り替わります。"
        ),
        "hook_patterns_none": "(GEMINI_API_KEY を設定すると抽出されます)",
        "formats": "フォーマット（頻度順）",
        "formats_none": "_(GEMINI_API_KEY を設定すると抽出されます)_",
        "vocabulary": "頻出ボキャブラリー",
        "vocabulary_none": "_(該当なし)_",
        "emotional_tone": "感情トーン",
        "emotional_tone_none": "_(GEMINI_API_KEY を設定すると抽出されます)_",
        "application": "自社ブランドへの応用",
        "application_none": "(GEMINI_API_KEY を設定すると生成されます)",
        "angle": "切り口",
        "hook_example": "フック例",
        "caveat_prefix": "注記",
        "others_header": "対象外クラスタ（言語インサイトは生成せず）",
        "others_row": "- クラスタ #{cid} ({n}本, 関連度 {score}/100): {label} — {reason}",
        "hook_pattern_row": "- `{pattern}` — 例: 「{example}」",
        # --- sound trends ---
        "sounds_header": "サウンドの傾向",
        "sounds_none": "複数クリエイターに使い回されているサウンドは見つかりませんでした。",
        "sounds_meta": "{n}本にサウンドID付き · オリジナル音声 {orig}%",
        "sounds_row": "**{name}** [{kind}] — {posts}本 / {creators}人{views}",
        "sounds_kind_original": "オリジナル",
        "sounds_kind_licensed": "既存/ライセンス",
        # --- self-audit scoreboard ---
        "score_header": "スコアボード",
        "score_clips": "診断した投稿",
        "score_high": "優先度・高",
        "score_overall": "全体所見",
        "score_recs": "横断的な改善アクション",
        "score_vocab_gaps": "共通して不足している語彙（{n}本中）",
        "score_sound": "自社のサウンド使用: {n}本中オリジナル音声 {orig}% · 既存サウンド {reused}種",
        "score_engagement": "実エンゲージメント: {n}本 · 中央値 {median}再生 · 最高 {hi} ({hi_v}再生) · 最低 {lo} ({lo_v}再生)",
        # --- self-audit tab ---
        "tab_insight": "① 業界を調べる",
        "tab_audit": "② 自分の投稿を診断",
        "insight_history_title": "📜 過去の分析結果を見る（API消費なし）",
        "insight_history_pick": "保存済みの分析結果（直近{keep}件）",
        "insight_history_load": "この結果を表示",
        "insight_history_load_failed": "読み込みに失敗しました（ファイルが壊れているか見つかりません）。",
        "audit_intro": (
            "**ステップ②です。** 自社ブランドの投稿を入れると、①で調べた"
            "業界パターンと比べて、フックの弱い点・取り入れるべき言い回し・"
            "フォーマットの改善案を出します。まず①を実行して基準を作ってください。"
        ),
        "audit_need_key": "GEMINI_API_KEYが未設定（またはサイドバーの「AIで詳しく分析する」がOFF）のため使えません。",
        "audit_mode": "投稿の入れ方",
        "audit_mode_upload": "動画をアップロード",
        "audit_mode_api": "Apify APIで取得",
        "audit_api_target": "Instagram プロフィール（例: your_brand）または投稿URL（カンマ区切り）",
        "audit_api_note": (
            "API取得だけではキャプション・ハッシュタグ・再生数・サウンドしか取れず、"
            "動画本体は見ません。下の「動画の中身も見る」をONにすると、指定した本数"
            "まで実際に動画をダウンロードしてGeminiに解析させます（話しているフック・"
            "画面内テキスト・フォーマットまで読み取ります）。"
        ),
        "audit_api_deep": "🎬 動画の中身も見る",
        "audit_api_deep_help": (
            "写真投稿を除く動画から指定本数まで、実際にファイルをダウンロードして"
            "中身を読みます。デフォルトは無料のローカル解析（faster-whisper＋"
            "EasyOCR＋PySceneDetect）でAPIは消費しません。"
        ),
        "audit_api_deep_limit": "解析する本数の上限",
        "audit_api_deep_limit_help": "再生数が多い投稿から優先的に解析します。残りはキャプションのみの簡易分析になります。",
        "audit_gemini_estimate": "🔢 この実行で消費するGemini API呼び出しの目安: 最大{n}回",
        "audit_api_deepened": "{n}本の動画を実際に解析しました。",
        "audit_api_deepen_failed": "{n}本は動画の解析に失敗しました（キャプションのみの分析で続行）。",
        "deep_video_none_found": (
            "「動画の中身も見る」はONでしたが、対象になった投稿がすべて写真投稿"
            "（スライドショー）または動画への直リンクが取得できないものだったため、"
            "実際に解析できた動画は0本でした。"
        ),
        "audit_api_run": "取得して診断",
        "audit_api_spinner": "Apifyで取得中…",
        "audit_api_deep_spinner": "Apifyで取得＋動画を解析中…（時間がかかります）",
        "upload_zone_title": "📤 動画をアップロード",
        "upload_zone_hint": "投稿した動画をドラッグ＆ドロップ、またはクリックして選択（mp4 / mov / m4v、複数可）",
        "audit_uploader": "投稿動画（mp4、複数可）",
        "audit_caption": "キャプション（任意・貼るほど精度が上がります）",
        "audit_post_url": "投稿URL（任意）",
        "audit_post_url_help": "入れると、この投稿の実際の再生数・いいね数・保存数をApifyで自動取得して診断に使います。",
        "audit_post_url_no_token": "実エンゲージメントの自動取得には .env の APIFY_API_TOKEN が必要です。",
        "audit_views_disabled": "投稿URLがあるので自動取得した実データを使います。",
        "audit_engagement_matched": "{n}本の投稿で実エンゲージメント（再生数・いいね等）を取得しました。",
        "audit_offline_note": "🔌 Geminiキー未設定のため、この実行はAPIを消費しません（機械的な簡易分析のみ）。",
        "audit_upload_gemini": "Geminiで詳しく解析する（動画1本＝API 1回消費）",
        "audit_upload_gemini_help": (
            "OFFの場合は無料のローカル解析（faster-whisperで文字起こし、EasyOCRで"
            "画面内テキスト、PySceneDetectでカット検出）のみを使い、Geminiは消費"
            "しません。ONにするとGeminiが動画を直接見て、ビジュアルタグや、より"
            "正確なフォーマット判定も得られますが、1本ごとにAPIを1回消費します。"
        ),
        "audit_api_deep_gemini": "Geminiで詳しく解析する（動画1本＝API 1回消費）",
        "audit_api_deep_gemini_help": (
            "OFFなら無料のローカル解析のみでAPIは消費しません。ONにするとGeminiが"
            "動画を直接見ます（視覚タグ・より正確なフォーマット判定つき）。"
        ),
        "niche_deep_gemini": "Geminiで詳しく解析する（動画1本＝API 1回消費）",
        "niche_deep_gemini_help": (
            "OFFなら無料のローカル解析のみでAPIは消費しません。ONにするとGeminiが"
            "動画を直接見ます（視覚タグ・より正確なフォーマット判定つき）。"
        ),
        "audit_history_title": "📜 過去の診断結果を見る（API消費なし）",
        "audit_history_pick": "保存済みの診断結果（直近{keep}件）",
        "audit_history_row": "{when} 保存 · {mode} · {n}本",
        "audit_history_load": "この結果を表示",
        "audit_history_load_failed": "読み込みに失敗しました（ファイルが壊れているか見つかりません）。",
        "audit_sort": "並び替え",
        "audit_sort_priority": "優先度順",
        "audit_sort_views": "再生数順",
        "audit_sort_likes": "いいね順",
        "audit_engagement_fetch_failed": "実エンゲージメントの取得に失敗しました（自己申告の数値のみで診断を続けます）",
        "audit_views": "再生数（任意・投稿URLがあれば不要）",
        "audit_run": "この動画を診断",
        "audit_spinner": "動画を解析中…（1本あたり数十秒）",
        "audit_no_ref": (
            "先に「① 業界を調べる」タブで分析を実行してください（診断の比較基準になります）。"
        ),
        "audit_building_ref": "比較基準（①の分析）を生成中…",
        "audit_ref_pick": "比較基準を選択（直近{keep}件まで保存）",
        "audit_ref_pick_row": "{when} 保存 · データ={src} · AI={gem} · 言語={lang}",
        "audit_ref_none": (
            "比較基準がまだありません。下のボタンで作成するか、「① 業界を調べる」タブで"
            "分析を実行してください。"
        ),
        "audit_ref_weak": "この比較基準はAIなしの簡易集計で作られています（フックの型・フォーマットが空）。「AIで詳しく分析する」をONにして作り直すと精度が上がります。",
        "audit_ref_lang": "この比較基準は別の言語（{ref}）で生成されています。診断結果は現在のUI言語で出しますが、精度を上げるには「比較基準をいま作成／作り直す」で作り直してください。",
        "audit_rebuild_ref": "比較基準をいま作成／作り直す",
        "audit_rebuilt": "基準を作り直して保存しました。",
        "audit_detected_hook": "検出したフック",
        "audit_detected_format": "検出したフォーマット",
        "audit_view_post": "投稿を見る",
        "audit_no_views_photo": "📷 写真投稿のため再生数なし",
        "audit_assessment": "総評",
        "audit_what_works": "できている点",
        "audit_gaps": "ギャップ",
        "audit_hook_rewrites": "フック書き換え案",
        "audit_rewrite_row": "- 現状: 「{original}」\n  → 改善: 「{rewrite}」\n  （基準: {based_on}）",
        "audit_format_suggestion": "フォーマット提案",
        "audit_vocab_adopt": "取り入れたい語彙",
        "audit_priority": "優先度",
        "audit_ref_hooks": "参照フックの型",
        "audit_error": "動画の解析に失敗しました",
        "audit_src_gemini": "🟢 Gemini",
        "audit_src_fallback": "⚪️ フォールバック（語彙差分のみ）",
    },
    "en": {
        "page_title": "Social Trend Agent - Language Insight",
        "hero_eyebrow": "SOCIAL TREND AGENT · SOCIAL LANGUAGE INTELLIGENCE",
        "title": "Job-Hunt Social Language Insight",
        "subtitle": (
            "Find the phrasing that's actually being used in job-hunt / career "
            "short-form video, then check your own posts against it."
        ),
        "how_title": "New here? (how it works)",
        "how_body": (
            "**Two steps.**\n\n"
            "1. **\"① Study the niche\" tab** — analyses a batch of job-hunt / "
            "career short-form videos and, per theme, pulls out the common hooks "
            "(opening line), video formats, phrasings and emotional tones. Just "
            "press **\"Run analysis\"** with the default settings.\n"
            "2. **\"② Audit your posts\" tab** — drop in your brand's own "
            "videos and it checks them against those niche patterns: weak hooks, "
            "phrasings to adopt, format tweaks.\n\n"
            "It only ever says a phrasing is *being used* — never that a change "
            "will go viral."
        ),
        "glossary_title": "Glossary",
        "glossary_body": (
            "- **Hook** — the first 2–3 seconds / opening line; it decides whether people keep watching.\n"
            "- **Hook pattern** — a fill-in-the-blank template like \"POV: {situation}\".\n"
            "- **Format** — how the video is made (talking head + captions / slideshow / green screen…).\n"
            "- **Cluster / theme** — a machine-grouped set of similar videos; one group = one topic.\n"
            "- **Relevance** — how close a group is to job-hunt / career talk (0–100).\n"
            "- **Insight** — the post-ready takeaways extracted for each theme."
        ),
        "advanced": "Advanced settings (usually leave as-is)",
        "no_engagement": (
            "This data has NO engagement (views / likes). Each phrasing can only "
            "be called \"being used\" in this niche — not trending, viral, or "
            "high-performing."
        ),
        "with_engagement": (
            "View / like counts are present, but this is a hashtag-search "
            "snapshot (not a time series), cumulative and not follower-"
            "normalised — read it as a relative signal within this sample, "
            "never \"trending right now\"."
        ),
        "language": "言語 / Language",
        "settings": "Settings",
        "data_source": "What to analyse",
        "source_json": "Try-it sample (runs instantly)",
        "source_hf": "Public dataset (real, large)",
        "source_apify": "Saved Apify files (real)",
        "source_apify_api": "Fetch now via Apify (real)",
        "apify_path": "Apify JSON path (file or directory)",
        "apify_platform": "Platform",
        "apify_hashtags": "Hashtags (comma-separated)",
        "apify_reels_only": "Reels (video) only",
        "apify_reels_only_help": (
            "OFF also fetches photo/slideshow posts, which have no video to "
            "watch (\"also watch the videos\" can't use them). ON restricts "
            "the fetch to Reels, so every post fetched is watchable."
        ),
        "apify_results_limit_hashtag": "Max videos per hashtag",
        "apify_results_limit_account": "Max videos per account",
        "apify_results_limit_help": (
            "This caps each hashtag, profile, or post URL individually, not the "
            "total. More of them means proportionally more rows fetched, more "
            "processing time, and more Apify usage (e.g. 2 hashtags × limit 100 "
            "→ up to 200 rows before de-dup)."
        ),
        "apify_no_token": "APIFY_API_TOKEN is not set in .env — cannot fetch.",
        "niche_deep": "🎬 Also watch the top-viewed videos",
        "niche_deep_help": (
            "Downloads up to N of the in-scope clips (highest views first) "
            "and actually reads them so hook/format/on-screen-text come from "
            "the video, not a guess from caption text. Default is free local "
            "reading (faster-whisper + EasyOCR + PySceneDetect) - no API spend."
        ),
        "niche_deep_limit": "Clips to deep-analyse",
        "niche_deep_limit_help": "The rest stay caption-only.",
        "niche_deep_tiktok_note": (
            "Watching video content currently only works with the Instagram "
            "fetch (the TikTok scraper doesn't return a direct video link)."
        ),
        "apify_need_query": "Enter at least one hashtag.",
        "data_file": "Data file",
        "median_views": "Median views",
        "matched_keywords": "Matched brand keywords",
        "download_json": "Download report.json",
        "download_md": "Download report.md",
        "download_audit_json": "Download audit results (JSON)",
        "download_audit_md": "Download audit results (Markdown)",
        "download_audit_pdf": "Download audit results (PDF)",
        "hf_limit": "Rows to fetch (Hugging Face)",
        "backend": "Text analysis method",
        "backend_help": "tfidf is offline and fast. sentence-transformers is more accurate but downloads a model on first use.",
        "n_clusters": "How many themes to find",
        "brand_profile": "Brand config file path",
        "relevance_threshold": "How tightly to filter to job-hunt / career (higher = stricter)",
        "use_gemini": "Analyse in depth with AI (recommended)",
        "use_gemini_help": (
            "On: Gemini generates hooks, formats and rewrite suggestions. "
            "Off: a simple word-frequency tally only (never calls the API even with a key)."
        ),
        "warn_no_key": "AI key (GEMINI_API_KEY) not set. Simple-tally mode only.",
        "info_offline": "Simple mode: no AI — word-frequency tally only.",
        "run": "Run analysis",
        "spinner": "Analyzing…",
        "error_prefix": "An error occurred while running",
        "prompt_run": (
            "**This is step ①.** Leave the settings as they are and click "
            "\"Run analysis\" to pull the phrasing patterns used in job-hunt social."
        ),
        "metric_clusters": "Themes found",
        "metric_videos": "Videos analyzed",
        "metric_targets": "Job-hunt / career themes",
        "no_targets": "No theme was close enough to job-hunt / career. Lower \"how tightly to filter\" in Advanced settings and re-run.",
        "src_gemini": "🟢 AI analysis",
        "src_fallback": "⚪️ simple tally",
        "cluster_meta": "{n} videos · job-hunt relevance {score}/100 · keywords: {label}",
        "theme_card_meta": "{n} videos · relevance {score}",
        "theme_grid_hint": "Click a theme to see its details below.",
        "examples": "Examples",
        "examples_top": "Examples (top by views)",
        "hook_patterns": "Hook patterns",
        "hook_patterns_video": "Hook patterns (video-verified)",
        "hook_patterns_caption": "Caption patterns (video not watched)",
        "hook_patterns_caption_note": (
            "These are inferred from caption text, not the video itself, and "
            "may not match what's actually said/shown as the hook. Turn on "
            "\"Also watch top-viewed videos\" to switch to hooks detected "
            "from actually-watched videos."
        ),
        "hook_patterns_none": "(set GEMINI_API_KEY to extract these)",
        "formats": "Formats (most frequent first)",
        "formats_none": "_(set GEMINI_API_KEY to extract these)_",
        "vocabulary": "Frequent vocabulary",
        "vocabulary_none": "_(none)_",
        "emotional_tone": "Emotional tone",
        "emotional_tone_none": "_(set GEMINI_API_KEY to extract this)_",
        "application": "Applying it to your brand",
        "application_none": "(set GEMINI_API_KEY to generate these)",
        "angle": "Angle",
        "hook_example": "Suggested hook",
        "caveat_prefix": "Caveat",
        "others_header": "Out-of-scope clusters (no insight generated)",
        "others_row": "- cluster #{cid} ({n} videos, relevance {score}/100): {label} — {reason}",
        "hook_pattern_row": "- `{pattern}` — e.g. “{example}”",
        # --- sound trends ---
        "sounds_header": "Sound trends",
        "sounds_none": "No sound was reused across multiple creators in this sample.",
        "sounds_meta": "{n} clips carry a sound id · original audio {orig}%",
        "sounds_row": "**{name}** [{kind}] — {posts} posts / {creators} creators{views}",
        "sounds_kind_original": "original",
        "sounds_kind_licensed": "licensed/other",
        # --- self-audit scoreboard ---
        "score_header": "Scoreboard",
        "score_clips": "Clips audited",
        "score_high": "High priority",
        "score_overall": "Overall read",
        "score_recs": "Cross-cutting actions",
        "score_vocab_gaps": "Vocabulary missing across clips (of {n})",
        "score_sound": "Your sound usage: original audio {orig}% of {n} · {reused} reused sounds",
        "score_engagement": "Real engagement: {n} clip(s) · median {median} views · best {hi} ({hi_v} views) · worst {lo} ({lo_v} views)",
        # --- self-audit tab ---
        "tab_insight": "① Study the niche",
        "tab_audit": "② Audit your posts",
        "insight_history_title": "📜 View past results (no API usage)",
        "insight_history_pick": "Saved results (last {keep} kept)",
        "insight_history_load": "View this result",
        "insight_history_load_failed": "Could not load it (file missing or corrupted).",
        "audit_intro": (
            "**This is step ②.** Drop in your brand's own posts and it checks "
            "them against the niche patterns from step ① — weak hooks, phrasings "
            "to adopt, format tweaks. Run step ① first to build the reference."
        ),
        "audit_need_key": "Needs GEMINI_API_KEY set (and \"Analyse in depth with AI\" ON in the sidebar).",
        "audit_mode": "How to add posts",
        "audit_mode_upload": "Upload videos",
        "audit_mode_api": "Fetch via Apify API",
        "audit_api_target": "Instagram profile (e.g. your_brand) or post URLs (comma-separated)",
        "audit_api_note": (
            "API fetch alone only gets caption, hashtags, views and sound — "
            "never the video itself. Turn on \"Also watch the videos\" below "
            "to have Gemini actually download and watch up to N of them "
            "(spoken hook, on-screen text, format)."
        ),
        "audit_api_deep": "🎬 Also watch the videos",
        "audit_api_deep_help": (
            "Downloads up to N video posts (skipping photos) and actually "
            "reads them. Default is free local reading (faster-whisper + "
            "EasyOCR + PySceneDetect) - no API spend."
        ),
        "audit_api_deep_limit": "Clips to deep-analyse",
        "audit_api_deep_limit_help": "Highest-viewed posts first. The rest stay caption-only.",
        "audit_gemini_estimate": "🔢 Estimated Gemini API calls this run will use: up to {n}",
        "audit_api_deepened": "Actually watched {n} video(s).",
        "audit_api_deepen_failed": "{n} clip(s) failed video analysis (kept caption-only).",
        "deep_video_none_found": (
            "\"Also watch the videos\" was ON, but every in-scope post was a "
            "photo/slideshow (or had no direct video link), so 0 clips could "
            "actually be watched."
        ),
        "audit_api_run": "Fetch & audit",
        "audit_api_spinner": "Fetching from Apify…",
        "audit_api_deep_spinner": "Fetching from Apify + watching videos… (this takes a while)",
        "upload_zone_title": "📤 Upload your videos",
        "upload_zone_hint": "Drag & drop your posted videos here, or click to browse (mp4 / mov / m4v, multiple allowed)",
        "audit_uploader": "Posted videos (mp4, multiple allowed)",
        "audit_caption": "Caption (optional — more text, better audit)",
        "audit_post_url": "Post URL (optional)",
        "audit_post_url_help": "If set, real views/likes/saves for this exact post are pulled from Apify and used in the audit.",
        "audit_post_url_no_token": "Auto-fetching real engagement needs APIFY_API_TOKEN in .env.",
        "audit_views_disabled": "Post URL set — using the auto-fetched real figure instead.",
        "audit_engagement_matched": "Fetched real engagement (views, likes, ...) for {n} post(s).",
        "audit_offline_note": "🔌 No Gemini key set - this run spends no API calls (mechanical analysis only).",
        "audit_upload_gemini": "Analyse in depth with Gemini (1 clip = 1 API call)",
        "audit_upload_gemini_help": (
            "OFF: free local reading only (faster-whisper for speech, EasyOCR "
            "for on-screen text, PySceneDetect for cut detection) - no Gemini "
            "spend. ON: Gemini watches the video directly - also gets visual "
            "tags and a more judged format read, but 1 API call per clip."
        ),
        "audit_api_deep_gemini": "Analyse in depth with Gemini (1 clip = 1 API call)",
        "audit_api_deep_gemini_help": (
            "OFF: free local reading only, no API spend. ON: Gemini watches "
            "each video directly (visual tags + a more judged format read)."
        ),
        "niche_deep_gemini": "Analyse in depth with Gemini (1 clip = 1 API call)",
        "niche_deep_gemini_help": (
            "OFF: free local reading only, no API spend. ON: Gemini watches "
            "each video directly (visual tags + a more judged format read)."
        ),
        "audit_history_title": "📜 View past audits (no API usage)",
        "audit_history_pick": "Saved audits (last {keep} kept)",
        "audit_history_row": "{when} · {mode} · {n} clip(s)",
        "audit_history_load": "View this result",
        "audit_history_load_failed": "Could not load it (file missing or corrupted).",
        "audit_sort": "Sort by",
        "audit_sort_priority": "Priority",
        "audit_sort_views": "Views",
        "audit_sort_likes": "Likes",
        "audit_engagement_fetch_failed": "Could not fetch real engagement (continuing with self-reported numbers only)",
        "audit_views": "Views (optional — skip if you gave a post URL)",
        "audit_run": "Audit these videos",
        "audit_spinner": "Analysing videos… (tens of seconds each)",
        "audit_no_ref": "Run the \"① Study the niche\" tab first — it is the reference the audit compares against.",
        "audit_building_ref": "Building the reference (step ①)…",
        "audit_ref_pick": "Choose a reference (last {keep} kept)",
        "audit_ref_pick_row": "{when} · data={src} · AI={gem} · lang={lang}",
        "audit_ref_none": (
            "No reference yet. Build one with the button below, or run the "
            "\"① Study the niche\" tab."
        ),
        "audit_ref_weak": "This reference was built without AI (hook patterns / formats are empty). Turn on \"Analyse in depth with AI\" and rebuild for a stronger audit.",
        "audit_ref_lang": "This reference was generated in another language ({ref}). The audit output will still be in your current UI language, but rebuild it (\"Build / rebuild the reference now\") for best results.",
        "audit_rebuild_ref": "Build / rebuild the reference now",
        "audit_rebuilt": "Reference rebuilt and saved.",
        "audit_detected_hook": "Detected hook",
        "audit_detected_format": "Detected format",
        "audit_view_post": "View post",
        "audit_no_views_photo": "📷 photo post — no view count",
        "audit_assessment": "Assessment",
        "audit_what_works": "What works",
        "audit_gaps": "Gaps",
        "audit_hook_rewrites": "Hook rewrites",
        "audit_rewrite_row": "- Now: “{original}”\n  → Better: “{rewrite}”\n  (based on: {based_on})",
        "audit_format_suggestion": "Format suggestion",
        "audit_vocab_adopt": "Vocabulary to adopt",
        "audit_priority": "Priority",
        "audit_ref_hooks": "Reference hook templates",
        "audit_error": "Video analysis failed",
        "audit_src_gemini": "🟢 Gemini",
        "audit_src_fallback": "⚪️ fallback (vocabulary diff only)",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "ja"

st.set_page_config(
    page_title=T[st.session_state.lang]["page_title"], page_icon="\U0001F4C8", layout="wide"
)

# --- Visual polish --------------------------------------------------------------
_STYLE = """
<style>
:root{
  --bg:#F7F8FA; --surface:#FFFFFF; --ink:#1A1D29; --muted:#6B7280;
  --line:#E5E7EB; --accent:#2F5CFF; --accent-ink:#1E3FCC;
  --shadow:0 1px 2px rgba(16,24,40,.04),0 8px 24px rgba(16,24,40,.05);
}
.stApp{ background:var(--bg); }
.block-container{max-width:1080px; padding-top:2.6rem; padding-bottom:4rem;}
html, body, [class*="css"]{ -webkit-font-smoothing:antialiased; }
h1,h2,h3,h4{ letter-spacing:.01em; font-weight:700; color:var(--ink); }

/* hero */
.hero{
  border:1px solid var(--line); border-radius:12px; padding:44px 48px;
  background:var(--surface);
  box-shadow:var(--shadow); margin-bottom:30px;
}
.hero .eyebrow{
  font-size:.68rem; letter-spacing:.24em; text-transform:uppercase;
  color:var(--accent); font-weight:700;
}
.hero h1{
  margin:.7rem 0 0; font-size:2rem; line-height:1.28; font-weight:700; color:var(--ink);
}
.hero .rule{ width:44px; height:3px; background:var(--accent); margin:20px 0; border-radius:2px; }
.hero p{ margin:0; color:var(--muted); font-size:.98rem; line-height:1.7; max-width:64ch; }

/* cards (st.container(border=True)) */
[data-testid="stVerticalBlockBorderWrapper"]{
  border:1px solid var(--line)!important; border-radius:12px!important;
  background:var(--surface); box-shadow:var(--shadow);
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"]{
  box-shadow:none;
}

/* metrics as quiet tiles */
[data-testid="stMetric"]{
  background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:16px 18px;
}
[data-testid="stMetricValue"]{ font-weight:700; color:var(--accent); }
[data-testid="stMetricLabel"] p{ color:var(--muted); font-size:.8rem; letter-spacing:.03em; }

/* buttons */
.stButton>button, .stDownloadButton>button{
  border-radius:8px; font-weight:600; border:1px solid var(--line);
  background:var(--surface); color:var(--ink); transition:all .15s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover{ border-color:var(--accent); color:var(--accent); }
.stButton>button[kind="primary"]{
  border:1px solid var(--accent); background:var(--accent); color:#fff;
  box-shadow:none; letter-spacing:.02em;
}
.stButton>button[kind="primary"]:hover{ background:var(--accent-ink); border-color:var(--accent-ink); color:#fff; }

/* tabs — underline style */
[data-testid="stTabs"] [data-baseweb="tab-list"]{ gap:28px; border-bottom:1px solid var(--line); }
[data-testid="stTabs"] button[role="tab"]{
  font-weight:600; font-size:1rem; padding:10px 0; color:var(--muted);
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{ background:var(--accent); }
[data-testid="stTabs"] button[aria-selected="true"]{ color:var(--accent); }

/* expander */
[data-testid="stExpander"]{
  border:1px solid var(--line); border-radius:12px; background:var(--surface); box-shadow:var(--shadow);
}
[data-testid="stExpander"] summary{ font-weight:600; }

/* sidebar */
[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--line); }

/* pills — thin, muted chips */
.pill{
  display:inline-block; padding:2px 10px; border-radius:999px; font-size:.72rem;
  font-weight:600; letter-spacing:.02em; border:1px solid var(--line);
  color:var(--muted); background:#F7F8FA; vertical-align:middle;
}
.pill.hi{ color:#B4232C; border-color:#F3D4D6; background:#FDF0F1; }
.pill.md{ color:#96660A; border-color:#F1E1B8; background:#FBF3DF; }
.pill.lo{ color:#0F7A50; border-color:#C9EBDC; background:#E9F9F1; }
.pill.ai{ color:var(--accent); border-color:#D6DFFF; background:#EEF2FF; }

/* upload / input zone (self-audit tab) */
.st-key-upload_zone{
  border:1px dashed #C7D1FF; border-radius:12px; background:#F7F9FF;
  padding:24px; margin-bottom:14px;
}
.upload-zone-title{ font-size:1.05rem; font-weight:700; color:var(--ink); margin:0 0 4px; }
.upload-zone-hint{ color:var(--muted); font-size:.86rem; margin:0 0 14px; }
.st-key-upload_zone [data-testid="stFileUploaderDropzone"]{
  background:var(--surface); border:1px solid var(--line); border-radius:10px;
}
.st-key-upload_zone [data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--accent); }
[data-testid="stFileUploaderDropzoneInstructions"] svg{ fill:var(--accent); }

/* per-file review card */
[data-testid="stExpander"] [data-testid="stExpander"]{ box-shadow:none; }
</style>
"""
st.markdown(_STYLE, unsafe_allow_html=True)

# Language picker at the very top of the sidebar so it applies before anything
# else renders.
with st.sidebar:
    st.session_state.lang = st.radio(
        T[st.session_state.lang]["language"],
        options=["ja", "en"],
        format_func=lambda x: "日本語" if x == "ja" else "English",
        index=["ja", "en"].index(st.session_state.lang),
        horizontal=True,
    )

lang = st.session_state.lang
t = T[lang]

st.markdown(
    f"""<div class="hero">
      <div class="eyebrow">{t["hero_eyebrow"]}</div>
      <h1>{t["title"]}</h1>
      <div class="rule"></div>
      <p>{t["subtitle"]}</p>
    </div>""",
    unsafe_allow_html=True,
)

with st.expander(t["how_title"], expanded="how_seen" not in st.session_state):
    st.markdown(t["how_body"])
st.session_state.how_seen = True
with st.expander(t["glossary_title"]):
    st.markdown(t["glossary_body"])

_SOURCE_LABELS = {
    "json": t["source_json"], "huggingface": t["source_hf"],
    "apify": t["source_apify"], "apify_api": t["source_apify_api"],
}

with st.sidebar:
    st.header(t["settings"])
    source = st.radio(
        t["data_source"],
        options=["json", "huggingface", "apify", "apify_api"],
        format_func=lambda x: _SOURCE_LABELS[x],
    )
    hf_limit = 1000
    apify_api_query = None
    deep_video_limit = 0
    deep_video_use_gemini = False
    if source == "json":
        data_path = st.text_input(t["data_file"], value=config.DATA_PATH)
    elif source == "apify":
        data_path = st.text_input(t["apify_path"], value=config.APIFY_PATH)
    elif source == "apify_api":
        data_path = config.DATA_PATH
        if not config.APIFY_API_TOKEN:
            st.warning(t["apify_no_token"])
        api_platform = st.radio(
            t["apify_platform"], options=["tiktok", "instagram"], horizontal=True,
        )
        api_reels_only = False
        if api_platform == "instagram":
            api_reels_only = st.checkbox(t["apify_reels_only"], value=False,
                                          help=t["apify_reels_only_help"])
        api_hashtags = st.text_input(t["apify_hashtags"], value="jobsearch, jobhunt")
        api_limit = st.number_input(
            t["apify_results_limit_hashtag"], min_value=10, max_value=1000,
            value=config.APIFY_RESULTS_LIMIT, step=10,
            help=t["apify_results_limit_help"],
        )
        api_tags = [h.strip().lstrip("#") for h in api_hashtags.split(",") if h.strip()]
        apify_api_query = {
            "platform": api_platform,
            "hashtags": api_tags,
            "results_limit": int(api_limit),
            "reels_only": api_reels_only,
        }
        if api_platform == "instagram":
            niche_deep_watch = st.checkbox(
                t["niche_deep"], value=False, help=t["niche_deep_help"],
            )
            if niche_deep_watch:
                deep_video_limit = int(st.number_input(
                    t["niche_deep_limit"], min_value=1, max_value=50,
                    value=5, step=1, help=t["niche_deep_limit_help"],
                ))
                deep_video_use_gemini = st.checkbox(
                    t["niche_deep_gemini"], value=False,
                    disabled=not config.GEMINI_API_KEY,
                    help=t["niche_deep_gemini_help"] if config.GEMINI_API_KEY
                    else t["warn_no_key"],
                )
                if config.GEMINI_API_KEY:
                    gemini_calls = 1 + (deep_video_limit if deep_video_use_gemini else 0)
                    st.caption(t["audit_gemini_estimate"].format(n=gemini_calls))
                else:
                    st.caption(t["audit_offline_note"])
        else:
            st.info(t["niche_deep_tiktok_note"])
    else:
        data_path = config.DATA_PATH
        hf_limit = st.number_input(
            t["hf_limit"], min_value=50, max_value=5000, value=300, step=50,
        )
    use_gemini = st.checkbox(
        t["use_gemini"], value=bool(config.GEMINI_API_KEY),
        help=t["use_gemini_help"],
    )

    with st.expander(t["advanced"]):
        backend = st.selectbox(
            t["backend"],
            options=["sentence-transformers", "tfidf"],
            index=["sentence-transformers", "tfidf"].index(config.EMBEDDING_BACKEND)
            if config.EMBEDDING_BACKEND in ("sentence-transformers", "tfidf") else 0,
            help=t["backend_help"],
        )
        n_clusters = st.slider(
            t["n_clusters"], min_value=2, max_value=15, value=config.N_CLUSTERS,
        )
        relevance_threshold = st.slider(
            t["relevance_threshold"],
            min_value=0, max_value=100, value=config.RELEVANCE_THRESHOLD,
        )
        brand_path = st.text_input(t["brand_profile"], value=config.BRAND_PATH)

    if use_gemini and not config.GEMINI_API_KEY:
        st.warning(t["warn_no_key"])
    elif not use_gemini:
        st.info(t["info_offline"])

    run_clicked = st.button(t["run"], type="primary", use_container_width=True)

if "report" not in st.session_state:
    st.session_state.report = None

# Hydrate the sound trends from the newest saved reference snapshot so the
# section shows even before a run in this session.
if "sounds" not in st.session_state:
    _latest = exporter.list_references(config.REFERENCE_HISTORY_DIR)
    _sounds = None
    if _latest:
        _, _ref_info = exporter.load_reference(_latest[0]["path"])
        _sounds = (_ref_info or {}).get("sounds") or None
    st.session_state.sounds = _sounds

if run_clicked:
    if source == "apify_api" and not apify_api_query["hashtags"]:
        st.error(t["apify_need_query"])
    else:
        with st.spinner(t["spinner"]):
            original_key = config.GEMINI_API_KEY
            if not use_gemini:
                config.GEMINI_API_KEY = ""
            try:
                collect = {}
                st.session_state.report = run(
                    data_path, n_clusters, backend, brand_path, relevance_threshold,
                    source, int(hf_limit), apify_api_query=apify_api_query,
                    collect=collect, language=lang, deep_video_limit=deep_video_limit,
                    deep_video_use_gemini=deep_video_use_gemini,
                )
                dv = collect.get("deep_video", {})
                if dv.get("n_deepened"):
                    st.success(t["audit_api_deepened"].format(n=dv["n_deepened"]))
                elif deep_video_limit > 0 and not dv.get("errors"):
                    st.info(t["deep_video_none_found"])
                if dv.get("errors"):
                    st.warning(t["audit_api_deepen_failed"].format(n=len(dv["errors"])))
                import sound_trends
                st.session_state.sounds = sound_trends.analyse_sounds(
                    collect.get("videos", []),
                    collect.get("engagement_available", False),
                    language=lang,
                )
                new_ref_path = exporter.save_reference(
                    st.session_state.report, config.REFERENCE_HISTORY_DIR,
                    meta={
                        "source": source, "n_clusters": n_clusters,
                        "relevance_threshold": relevance_threshold,
                        "gemini": bool(config.GEMINI_API_KEY), "language": lang,
                    },
                    sounds=st.session_state.sounds,
                    keep=config.REFERENCE_HISTORY_KEEP,
                )
                st.session_state.ref_choice_path = new_ref_path
            except Exception as e:  # noqa: BLE001
                st.session_state.report = None
                st.error(f"{t['error_prefix']}: {type(e).__name__}: {e}")
            finally:
                config.GEMINI_API_KEY = original_key

report = st.session_state.report

tab_insight, tab_audit = st.tabs([t["tab_insight"], t["tab_audit"]])


def render_insight_history_picker():
    """Past niche runs, saved by every "Run analysis" to REFERENCE_HISTORY_DIR.
    Loading one from here is free (no Apify / Gemini call) - use it to revisit
    a past result instead of always re-running and re-spending the API calls.
    """
    choices = exporter.list_references(config.REFERENCE_HISTORY_DIR)
    if not choices:
        return
    with st.expander(t["insight_history_title"], expanded=False):
        paths = [c["path"] for c in choices]

        def _fmt(p: str) -> str:
            c = next(c for c in choices if c["path"] == p)
            m = c["meta"]
            return t["audit_ref_pick_row"].format(
                when=c["saved_at"], src=m.get("source", "?"),
                gem="✓" if m.get("gemini") else "✗", lang=m.get("language", "?"),
            )

        sel = st.selectbox(
            t["insight_history_pick"].format(keep=config.REFERENCE_HISTORY_KEEP),
            options=paths, format_func=_fmt, key="insight_hist_pick",
        )
        if st.button(t["insight_history_load"], key="insight_hist_load_btn"):
            rep, info = exporter.load_reference(sel)
            if rep is None:
                st.error(t["insight_history_load_failed"])
            else:
                st.session_state.report = rep
                st.session_state.sounds = (info or {}).get("sounds") or None
                st.session_state.ref_choice_path = sel
                st.rerun()


_THEME_CARD_CSS = """
<style>
div[class*="st-key-theme_card_"] button {
    aspect-ratio: 1 / 1;
    width: 100%;
    height: auto;
    white-space: normal;
    font-weight: 600;
    font-size: 0.95rem;
    line-height: 1.3;
}
</style>
"""


def _render_theme_grid(targets):
    st.markdown(_THEME_CARD_CSS, unsafe_allow_html=True)
    st.caption(t["theme_grid_hint"])
    if "selected_theme_idx" not in st.session_state:
        st.session_state.selected_theme_idx = 0
    for row_start in range(0, len(targets), 3):
        row = list(enumerate(targets))[row_start:row_start + 3]
        cols = st.columns(3)
        for col, (i, c) in zip(cols, row):
            ins = c["insight"]
            with col:
                is_selected = st.session_state.selected_theme_idx == i
                if st.button(
                    f"{i + 1}. {ins['theme']}",
                    key=f"theme_card_{c['cluster_id']}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                ):
                    st.session_state.selected_theme_idx = i
                    st.rerun()
                st.caption(t["theme_card_meta"].format(
                    n=c["size"], score=c["relevance_score"],
                ))


def _render_theme_detail(c, i, engagement_available):
    ins = c["insight"]
    is_ai = ins["source"] == "gemini"
    src_pill = (
        f'<span class="pill ai">{t["src_gemini"]}</span>' if is_ai
        else f'<span class="pill">{t["src_fallback"]}</span>'
    )
    with st.container(border=True):
        st.markdown(
            f"<h3 style='margin:.1rem 0'>{i}. {ins['theme']} &nbsp;{src_pill}</h3>",
            unsafe_allow_html=True,
        )
        meta = t["cluster_meta"].format(
            cid=c["cluster_id"], n=c["size"], score=c["relevance_score"],
            label=c["label"],
        )
        if c["median_views"]:
            meta += f" · {t['median_views']} {c['median_views']:,}"
        st.caption(meta)
        if c["distinct_keywords"]:
            st.caption(f"{t['matched_keywords']}: " + ", ".join(c["distinct_keywords"]))

        st.markdown(f"**{t['examples_top'] if engagement_available else t['examples']}**")
        for ex in c["examples"]:
            quote = ex["on_screen_text"] or ex["caption"]
            quote = (quote[:140] + "…") if len(quote) > 141 else quote
            views = f" · {ex['views']:,} views" if ex["views"] else ""
            link = f" ([link]({ex['url']}))" if ex["url"] else ""
            st.markdown(f"- 「{quote}」 (@{ex['creator']}){views}{link}")

        hook_source = ins.get("hook_source", "caption")
        hook_label = (
            t["hook_patterns_video"] if hook_source == "video"
            else t["hook_patterns_caption"] if hook_source == "caption"
            else t["hook_patterns"]
        )
        st.markdown(f"**{hook_label}**")
        if ins["hook_patterns"]:
            for hp in ins["hook_patterns"]:
                st.markdown(t["hook_pattern_row"].format(
                    pattern=hp["pattern"], example=hp["example"],
                ))
            if hook_source == "caption":
                st.caption(t["hook_patterns_caption_note"])
        else:
            st.caption(t["hook_patterns_none"])

        st.markdown(f"**{t['formats']}**")
        st.markdown(", ".join(ins["formats"]) if ins["formats"] else t["formats_none"])

        st.markdown(f"**{t['vocabulary']}**")
        st.markdown(", ".join(ins["vocabulary"]) if ins["vocabulary"] else t["vocabulary_none"])

        st.markdown(f"**{t['emotional_tone']}**")
        st.markdown(
            ", ".join(ins["emotional_tone"]) if ins["emotional_tone"]
            else t["emotional_tone_none"]
        )

        st.markdown(f"**{t['application']}**")
        if ins["brand_application"]:
            for ap in ins["brand_application"]:
                tool = f"[{ap['tool']}] " if ap.get("tool") else ""
                st.markdown(f"- {tool}**{t['angle']}**: {ap['angle']}")
                if ap["suggested_hook"]:
                    st.markdown(f"  **{t['hook_example']}**: 「{ap['suggested_hook']}」")
        else:
            st.caption(t["application_none"])

        st.info(f"{t['caveat_prefix']}: {ins['caveat']}")


def render_insight_tab(report):
    render_insight_history_picker()
    if report is None:
        st.info(t["prompt_run"])
        return
    targets = [c for c in report if c["is_target"]]
    others = [c for c in report if not c["is_target"]]
    engagement_available = any(
        ex["views"] for c in report for ex in c["examples"]
    )
    st.warning(t["with_engagement"] if engagement_available else t["no_engagement"])

    col1, col2, col3 = st.columns(3)
    col1.metric(t["metric_clusters"], len(report))
    col2.metric(t["metric_videos"], sum(c["size"] for c in report))
    col3.metric(t["metric_targets"], len(targets))

    dl1, dl2 = st.columns(2)
    dl1.download_button(t["download_json"], exporter.report_json(report),
                        file_name="report.json", mime="application/json")
    dl2.download_button(t["download_md"], exporter.report_markdown(report),
                        file_name="report.md", mime="text/markdown")

    if not targets:
        st.info(t["no_targets"])

    if targets:
        _render_theme_grid(targets)
        st.divider()
        sel = min(st.session_state.get("selected_theme_idx", 0), len(targets) - 1)
        _render_theme_detail(targets[sel], sel + 1, engagement_available)

    _render_sound_trends(st.session_state.get("sounds"))

    if others:
        st.divider()
        st.markdown(f"**{t['others_header']}**")
        for c in others:
            st.markdown(t["others_row"].format(
                cid=c["cluster_id"], n=c["size"], score=c["relevance_score"],
                label=c["label"], reason=c["exclude_reason"],
            ))


def _render_sound_trends(sounds):
    if not sounds or not sounds.get("n_with_sound"):
        return
    st.divider()
    st.markdown(f"**{t['sounds_header']}**")
    st.caption(t["sounds_meta"].format(
        n=sounds["n_with_sound"], orig=sounds.get("original_share_pct", "?"),
    ))
    rows = sounds.get("sounds", [])
    if not rows:
        st.caption(t["sounds_none"])
        return
    for s in rows:
        kind = t["sounds_kind_original"] if s["is_original"] else t["sounds_kind_licensed"]
        views = f" · {s['median_views']:,} med. views" if s.get("median_views") else ""
        st.markdown(t["sounds_row"].format(
            name=s["sound_name"], kind=kind, posts=s["post_count"],
            creators=s["creator_count"], views=views,
        ))
        if s["hashtags"]:
            st.caption("· ".join(s["hashtags"]))
    st.info(f"{t['caveat_prefix']}: {sounds.get('caveat', '')}")


# --- Self-audit tab -------------------------------------------------------------

_PRIORITY_PILL = {"high": "hi", "medium": "md", "low": "lo"}


def _pill(text: str, kind: str = "") -> str:
    cls = f"pill {kind}".strip()
    return f'<span class="{cls}">{text}</span>'


def _gemini_call_estimate(n_video_reads: int, has_ref: bool) -> int:
    """Upper bound on Gemini calls ONE audit run will spend: 1 to (re)build
    the reference if it isn't cached, 1 to score the whole batch (always
    batched - never per clip), 1 for the portfolio rollup (skipped below 2
    clips, but this is a worst-case upper bound), plus 1 per video actually
    watched (analyse_video_file has no batched form - it's 1 call each)."""
    return (0 if has_ref else 1) + 1 + 1 + max(0, n_video_reads)


def _audit_uploaded_videos(files, meta, report, use_gemini: bool = False):
    """Analyse each uploaded clip, then audit the batch against the niche
    reference. `meta[name]` -> {"caption": str, "views": int, "url": str}.

    `use_gemini=False` (default): reads each clip locally via video_local -
    free, no daily-quota risk, mechanical format guess, no visual_tags.
    `use_gemini=True`: sends each to Gemini's File API instead - 1 call per
    clip, also gets visual_tags and a genuinely judged format.

    When a clip's `url` is filled in AND an Apify token is configured, fetches
    REAL engagement (views/likes/comments/saves + sound) for exactly those
    post URLs in one Apify call and merges it in - pixel-derived fields
    (hook, format, transcript, ...) stay from whichever reader ran, either way.

    Returns (own_videos, audit_items, n_engagement_matched) or raises on a
    per-clip analysis failure.
    """
    import tempfile

    from brand_match import load_brand
    from ingest import load_videos_from_apify_api
    import video_audit

    if use_gemini:
        reader = video_audit.analyse_video_file
    else:
        from video_local import analyse_video_file_locally as reader

    own_videos = []
    for f in files:
        suffix = os.path.splitext(f.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(f.getbuffer())
            tmp_path = tmp.name
        try:
            m = meta.get(f.name, {})
            own_videos.append(reader(
                tmp_path, caption=m.get("caption", ""),
                views=int(m.get("views") or 0), video_id=f.name,
                url=m.get("url", ""),
            ))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    n_matched = 0
    post_urls = [v.url for v in own_videos if v.url]
    if post_urls and config.APIFY_API_TOKEN:
        try:
            apify_videos = load_videos_from_apify_api(
                platform="instagram", post_urls=post_urls,
                results_limit=len(post_urls),
                token=config.APIFY_API_TOKEN,
                instagram_actor=config.APIFY_INSTAGRAM_ACTOR,
                sync_timeout=config.APIFY_SYNC_TIMEOUT,
            )
            own_videos, n_matched = video_audit.merge_apify_engagement(
                own_videos, apify_videos,
            )
        except Exception as e:  # noqa: BLE001 - engagement is a bonus, never fatal
            st.warning(f"{t['audit_engagement_fetch_failed']}: {type(e).__name__}: {e}")

    brand = load_brand(brand_path)
    audit_items = video_audit.audit_against_reference(
        own_videos, report, brand, language=lang,
    )
    return own_videos, audit_items, n_matched


def _fetch_own_via_apify(
    target: str, limit: int, report, deep_limit: int = 0, deep_use_gemini: bool = False,
    reels_only: bool = False,
):
    """Pull the brand's own posts through the Apify API (Instagram) and audit
    them against the niche reference. `target` is a comma list of handles /
    profile URLs / post URLs.

    By default the clips carry caption + hashtags + engagement + sound only
    (Apify never decodes video frames). When `deep_limit` > 0, up to that many
    video posts (skipping photo/slideshow posts) are additionally downloaded
    from Apify's own `videoUrl` and actually read - spoken hook, on-screen
    text, format - same depth as the manual upload path, just picked by
    handle instead of by hand. `deep_use_gemini=False` (default) reads them
    locally (free, no quota risk); True sends each to Gemini's File API
    instead (1 call per clip, also gets visual_tags + a judged format).

    Returns (own_videos, audit_items, n_deepened, deepen_errors).
    """
    from brand_match import load_brand
    from ingest import load_videos_from_apify_api
    import video_audit

    items = [x.strip() for x in target.split(",") if x.strip()]
    urls = [x for x in items if x.startswith("http")]
    profiles = [x for x in items if not x.startswith("http")]
    own_videos = load_videos_from_apify_api(
        platform="instagram",
        profiles=profiles or None,
        post_urls=urls or None,
        results_limit=limit,
        token=config.APIFY_API_TOKEN,
        tiktok_actor=config.APIFY_TIKTOK_ACTOR,
        instagram_actor=config.APIFY_INSTAGRAM_ACTOR,
        sync_timeout=config.APIFY_SYNC_TIMEOUT,
        reels_only=reels_only,
    )

    n_deepened, deepen_errors = 0, []
    if deep_limit > 0 and (not deep_use_gemini or config.GEMINI_API_KEY):
        own_videos, n_deepened, deepen_errors = video_audit.deepen_with_video_content(
            own_videos, limit=deep_limit, use_gemini=deep_use_gemini,
        )

    brand = load_brand(brand_path)
    audit_items = video_audit.audit_against_reference(
        own_videos, report, brand, language=lang,
    )
    return own_videos, audit_items, n_deepened, deepen_errors


def _build_and_save_reference():
    """Run the niche pipeline with the current sidebar settings, cache it, and
    put it in session_state. Returns the report."""
    collect = {}
    rep = run(
        data_path, n_clusters, backend, brand_path, relevance_threshold,
        source, int(hf_limit), apify_api_query=apify_api_query,
        collect=collect, language=lang, deep_video_limit=deep_video_limit,
                    deep_video_use_gemini=deep_video_use_gemini,
    )
    dv = collect.get("deep_video", {})
    if dv.get("n_deepened"):
        st.success(t["audit_api_deepened"].format(n=dv["n_deepened"]))
    elif deep_video_limit > 0 and not dv.get("errors"):
        st.info(t["deep_video_none_found"])
    if dv.get("errors"):
        st.warning(t["audit_api_deepen_failed"].format(n=len(dv["errors"])))
    import sound_trends
    st.session_state.sounds = sound_trends.analyse_sounds(
        collect.get("videos", []),
        collect.get("engagement_available", False),
        language=lang,
    )
    new_ref_path = exporter.save_reference(
        rep, config.REFERENCE_HISTORY_DIR,
        meta={
            "source": source, "n_clusters": n_clusters,
            "relevance_threshold": relevance_threshold,
            "gemini": bool(config.GEMINI_API_KEY), "language": lang,
        },
        sounds=st.session_state.sounds,
        keep=config.REFERENCE_HISTORY_KEEP,
    )
    st.session_state.report = rep
    st.session_state.ref_choice_path = new_ref_path
    return rep


def _reference_is_weak(rep) -> bool:
    targets = [c for c in rep if c.get("is_target")]
    return bool(targets) and all(
        (c.get("insight") or {}).get("source") == "fallback" for c in targets
    )


def _render_scoreboard(summary):
    if not summary:
        return
    with st.container(border=True):
        st.subheader(t["score_header"])
        pc = summary.get("priority_counts", {})
        c1, c2 = st.columns(2)
        c1.metric(t["score_clips"], summary.get("n_clips", 0))
        c2.metric(t["score_high"], pc.get("high", 0))

        snd = summary.get("sound", {})
        if snd.get("n_with_sound"):
            st.caption(t["score_sound"].format(
                n=snd["n_with_sound"], orig=snd.get("original_share_pct", "?"),
                reused=snd.get("distinct_reused_sounds", 0),
            ))

        eng = summary.get("engagement", {})
        if eng.get("n_with_engagement"):
            st.caption(t["score_engagement"].format(
                n=eng["n_with_engagement"], median=f"{eng['median_views']:,}",
                hi=eng["highest"]["video_id"], hi_v=f"{eng['highest']['views']:,}",
                lo=eng["lowest"]["video_id"], lo_v=f"{eng['lowest']['views']:,}",
            ))

        if summary.get("overall"):
            st.markdown(f"**{t['score_overall']}**")
            st.write(summary["overall"])

        if summary.get("recommendations"):
            st.markdown(f"**{t['score_recs']}**")
            for r in summary["recommendations"]:
                st.markdown(f"- {r}")

        gaps = summary.get("top_vocabulary_gaps", [])
        if gaps:
            st.markdown(
                f"**{t['score_vocab_gaps'].format(n=summary.get('n_clips', 0))}**"
            )
            st.markdown(", ".join(f"{g['term']} ({g['count']})" for g in gaps))


def render_audit_tab():
    st.caption(t["audit_intro"])

    # Which reference will the audit score against? Up to REFERENCE_HISTORY_KEEP
    # past niche runs are kept on disk; the user picks one (newest by default -
    # a fresh sidebar run or "rebuild" sets ref_choice_path to it automatically).
    choices = exporter.list_references(config.REFERENCE_HISTORY_DIR)
    ref_report, ref_info = None, None

    if choices:
        paths = [c["path"] for c in choices]
        default_path = st.session_state.get("ref_choice_path")
        if default_path not in paths:
            default_path = paths[0]

        def _fmt_choice(p: str) -> str:
            c = next(c for c in choices if c["path"] == p)
            m = c["meta"]
            return t["audit_ref_pick_row"].format(
                when=c["saved_at"], src=m.get("source", "?"),
                gem="✓" if m.get("gemini") else "✗",
                lang=m.get("language", "?"),
            )

        ref_path = st.selectbox(
            t["audit_ref_pick"].format(keep=config.REFERENCE_HISTORY_KEEP),
            options=paths, format_func=_fmt_choice,
            index=paths.index(default_path),
        )
        st.session_state.ref_choice_path = ref_path
        ref_report, ref_info = exporter.load_reference(ref_path)
    else:
        st.info(t["audit_ref_none"])

    has_ref = ref_report is not None
    if has_ref:
        m = (ref_info or {}).get("meta", {})
        ref_lang = m.get("language")
        if ref_lang and ref_lang != lang:
            st.warning(t["audit_ref_lang"].format(ref=ref_lang))
        if _reference_is_weak(ref_report):
            st.warning(t["audit_ref_weak"])

    if st.button(
        t["audit_rebuild_ref"], use_container_width=True,
        type="primary" if not has_ref else "secondary",
    ):
        original_key = config.GEMINI_API_KEY
        if not use_gemini:
            config.GEMINI_API_KEY = ""
        try:
            with st.spinner(t["audit_building_ref"]):
                ref_report = _build_and_save_reference()
            st.success(t["audit_rebuilt"])
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"{t['error_prefix']}: {type(e).__name__}: {e}")
        finally:
            config.GEMINI_API_KEY = original_key

    # Past self-audit runs, saved by every completed audit to AUDIT_HISTORY_DIR.
    # Opening one here is free (no Apify / Gemini call) - use it to revisit a
    # past result instead of always re-running and re-spending the API calls.
    audit_choices = exporter.list_audits(config.AUDIT_HISTORY_DIR)
    if audit_choices:
        with st.expander(t["audit_history_title"], expanded=False):
            a_paths = [c["path"] for c in audit_choices]

            def _fmt_audit(p: str) -> str:
                c = next(c for c in audit_choices if c["path"] == p)
                m = c["meta"]
                return t["audit_history_row"].format(
                    when=c["saved_at"],
                    mode=t["audit_mode_upload"] if m.get("mode") == "upload"
                    else t["audit_mode_api"],
                    n=m.get("n_clips", "?"),
                )

            a_sel = st.selectbox(
                t["audit_history_pick"].format(keep=config.AUDIT_HISTORY_KEEP),
                options=a_paths, format_func=_fmt_audit, key="audit_hist_pick",
            )
            if st.button(t["audit_history_load"], key="audit_hist_load_btn"):
                a_items, a_videos, a_summary, _a_info = exporter.load_audit(a_sel)
                if a_items is None:
                    st.error(t["audit_history_load_failed"])
                else:
                    st.session_state.audit = {
                        "videos": a_videos, "items": a_items, "summary": a_summary,
                    }
                    st.rerun()

    mode = st.radio(
        t["audit_mode"], options=["upload", "api"], horizontal=True,
        format_func=lambda x: t["audit_mode_upload"] if x == "upload"
        else t["audit_mode_api"],
    )
    key_ready = use_gemini and bool(config.GEMINI_API_KEY)

    files, meta = None, {}
    api_target, api_limit, deep_limit = "", config.APIFY_RESULTS_LIMIT, 0
    run_audit = False

    if mode == "upload":
        upload_use_gemini = st.checkbox(
            t["audit_upload_gemini"], value=False, disabled=not key_ready,
            help=t["audit_upload_gemini_help"] if key_ready else t["audit_need_key"],
        )
        with st.container(key="upload_zone"):
            st.markdown(f'<p class="upload-zone-title">{t["upload_zone_title"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="upload-zone-hint">{t["upload_zone_hint"]}</p>', unsafe_allow_html=True)
            files = st.file_uploader(
                t["audit_uploader"], type=["mp4", "mov", "m4v"],
                accept_multiple_files=True, label_visibility="collapsed",
            )
        if files:
            for f in files:
                with st.expander(f.name, expanded=len(files) == 1):
                    cap = st.text_area(
                        t["audit_caption"], key=f"cap_{f.name}", height=80,
                    )
                    url = st.text_input(
                        t["audit_post_url"], key=f"url_{f.name}",
                        placeholder="https://www.instagram.com/reel/...",
                        help=t["audit_post_url_help"] if config.APIFY_API_TOKEN
                        else t["audit_post_url_no_token"],
                    )
                    vw = st.number_input(
                        t["audit_views"], min_value=0, value=0, step=100,
                        key=f"vw_{f.name}", disabled=bool(url.strip()),
                        help=t["audit_views_disabled"] if url.strip() else None,
                    )
                    meta[f.name] = {"caption": cap, "views": vw, "url": url.strip()}
        if files:
            if config.GEMINI_API_KEY:
                st.caption(t["audit_gemini_estimate"].format(
                    n=_gemini_call_estimate(len(files) if upload_use_gemini else 0, has_ref)
                ))
            else:
                st.caption(t["audit_offline_note"])
        run_audit = st.button(
            t["audit_run"], type="primary", disabled=not files,
            use_container_width=True,
        )
    else:
        st.info(t["audit_api_note"])
        if not config.APIFY_API_TOKEN:
            st.warning(t["apify_no_token"])
        api_target = st.text_input(t["audit_api_target"], value="")
        api_reels_only = st.checkbox(t["apify_reels_only"], value=False,
                                      help=t["apify_reels_only_help"])
        api_limit = st.number_input(
            t["apify_results_limit_account"], min_value=5, max_value=500,
            value=config.APIFY_RESULTS_LIMIT, step=5,
            help=t["apify_results_limit_help"],
        )
        deep_watch = st.checkbox(
            t["audit_api_deep"], value=False, help=t["audit_api_deep_help"],
        )
        deep_limit, deep_use_gemini = 0, False
        if deep_watch:
            deep_limit = int(st.number_input(
                t["audit_api_deep_limit"], min_value=1, max_value=100,
                value=5, step=1, help=t["audit_api_deep_limit_help"],
            ))
            deep_use_gemini = st.checkbox(
                t["audit_api_deep_gemini"], value=False, disabled=not key_ready,
                help=t["audit_api_deep_gemini_help"] if key_ready else t["audit_need_key"],
            )
        if config.GEMINI_API_KEY:
            st.caption(t["audit_gemini_estimate"].format(
                n=_gemini_call_estimate(deep_limit if deep_use_gemini else 0, has_ref)
            ))
        else:
            st.caption(t["audit_offline_note"])
        run_audit = st.button(
            t["audit_api_run"], type="primary",
            disabled=not (api_target.strip() and config.APIFY_API_TOKEN),
            use_container_width=True,
        )

    if run_audit:
        original_key = config.GEMINI_API_KEY
        try:
            if ref_report is None:
                with st.spinner(t["audit_building_ref"]):
                    ref_report = _build_and_save_reference()
            n_matched = 0
            if mode == "upload":
                with st.spinner(t["audit_spinner"]):
                    own_videos, audit_items, n_matched = _audit_uploaded_videos(
                        files, meta, ref_report, use_gemini=upload_use_gemini,
                    )
                if n_matched:
                    st.success(t["audit_engagement_matched"].format(n=n_matched))
            else:
                spinner_msg = (
                    t["audit_api_deep_spinner"] if deep_limit else t["audit_api_spinner"]
                )
                with st.spinner(spinner_msg):
                    own_videos, audit_items, n_deepened, deepen_errors = _fetch_own_via_apify(
                        api_target, int(api_limit), ref_report, deep_limit=deep_limit,
                        deep_use_gemini=deep_use_gemini, reels_only=api_reels_only,
                    )
                if n_deepened:
                    st.success(t["audit_api_deepened"].format(n=n_deepened))
                elif deep_limit > 0 and not deepen_errors:
                    st.info(t["deep_video_none_found"])
                if deepen_errors:
                    st.warning(
                        t["audit_api_deepen_failed"].format(n=len(deepen_errors))
                    )
            from brand_match import load_brand
            import video_audit
            summary = video_audit.summarise_audit(
                audit_items, own_videos, ref_report, load_brand(brand_path),
                language=lang,
            )
            audit_videos = {v.id: v for v in own_videos}
            st.session_state.audit = {
                "videos": audit_videos,
                "items": audit_items,
                "summary": summary,
            }
            try:
                exporter.save_audit(
                    audit_items, audit_videos, summary, config.AUDIT_HISTORY_DIR,
                    meta={
                        "mode": mode, "n_clips": len(own_videos),
                        "gemini": bool(config.GEMINI_API_KEY), "language": lang,
                    },
                    keep=config.AUDIT_HISTORY_KEEP,
                )
            except OSError:
                pass  # saving the history is a bonus, never fatal to the audit
        except Exception as e:  # noqa: BLE001
            st.session_state.audit = None
            st.error(f"{t['audit_error']}: {type(e).__name__}: {e}")
        finally:
            config.GEMINI_API_KEY = original_key

    audit = st.session_state.get("audit")
    if not audit:
        return

    _render_scoreboard(audit.get("summary"))

    adl1, adl2, adl3 = st.columns(3)
    adl1.download_button(
        t["download_audit_json"],
        exporter.audit_json(audit["items"], audit["videos"], audit.get("summary")),
        file_name="self_audit.json", mime="application/json",
    )
    adl2.download_button(
        t["download_audit_md"],
        exporter.audit_markdown(audit["items"], audit["videos"], audit.get("summary"), lang),
        file_name="self_audit.md", mime="text/markdown",
    )
    adl3.download_button(
        t["download_audit_pdf"],
        exporter.audit_pdf(audit["items"], audit["videos"], audit.get("summary"), lang),
        file_name="self_audit.pdf", mime="application/pdf",
    )

    _order = {"high": 0, "medium": 1, "low": 2}
    videos = audit["videos"]
    has_views = any(v.engagement.get("views") for v in videos.values())
    has_likes = any(v.engagement.get("likes") for v in videos.values())

    sort_options = ["priority"] + (["views"] if has_views else []) + (["likes"] if has_likes else [])
    if len(sort_options) > 1:
        sort_key = st.radio(
            t["audit_sort"], options=sort_options,
            format_func=lambda x: t[f"audit_sort_{x}"], horizontal=True,
        )
    else:
        sort_key = "priority"

    def _sort_value(it):
        if sort_key in ("views", "likes"):
            v = videos.get(it["video_id"])
            metric = v.engagement.get(sort_key, 0) if v else 0
            return (0, -metric)
        return (_order.get(it["priority"], 1), 0)

    for item in sorted(audit["items"], key=_sort_value):
        v = audit["videos"].get(item["video_id"])
        src = (
            t["audit_src_gemini"] if item.get("source") == "gemini"
            else t["audit_src_fallback"]
        )
        with st.container(border=True):
            pri = item["priority"]
            pills = (
                _pill(f"{t['audit_priority']}: {pri}", _PRIORITY_PILL.get(pri, ""))
                + " "
                + (_pill(src, "ai") if item.get("source") == "gemini" else _pill(src))
            )
            st.markdown(
                f"<h3 style='margin:.1rem 0'>{item['video_id']} &nbsp;{pills}</h3>",
                unsafe_allow_html=True,
            )

            if v is not None:
                hook_line = v.spoken_hook or (v.caption[:120] if v.caption else "")
                if hook_line:
                    st.markdown(f"**{t['audit_detected_hook']}**: 「{hook_line}」")
                if v.detected_format:
                    st.markdown(f"**{t['audit_detected_format']}**: {v.detected_format}")
                if v.url:
                    st.markdown(f"🔗 [{t['audit_view_post']}]({v.url})")
                bits = []
                if v.engagement.get("views"):
                    bits.append(f"{v.engagement['views']:,} views")
                elif v.is_slideshow and v.engagement_is_real:
                    bits.append(t["audit_no_views_photo"])
                if v.engagement.get("likes"):
                    bits.append(f"{v.engagement['likes']:,} likes")
                if v.sound_name:
                    bits.append(f"♪ {v.sound_name}")
                if v.visual_tags:
                    bits.append(", ".join(v.visual_tags))
                if bits:
                    st.caption(" · ".join(bits))

            if item["assessment"]:
                st.markdown(f"**{t['audit_assessment']}**")
                st.write(item["assessment"])

            if item["what_works"]:
                st.markdown(f"**{t['audit_what_works']}**")
                for x in item["what_works"]:
                    st.markdown(f"- {x}")

            if item["gaps"]:
                st.markdown(f"**{t['audit_gaps']}**")
                for x in item["gaps"]:
                    st.markdown(f"- {x}")

            if item["hook_rewrites"]:
                st.markdown(f"**{t['audit_hook_rewrites']}**")
                for rw in item["hook_rewrites"]:
                    st.markdown(t["audit_rewrite_row"].format(
                        original=rw["original"] or "—",
                        rewrite=rw["rewrite"],
                        based_on=rw["based_on"] or "—",
                    ))

            if item["format_suggestion"]:
                st.markdown(f"**{t['audit_format_suggestion']}**: {item['format_suggestion']}")

            if item["vocabulary_to_adopt"]:
                st.markdown(
                    f"**{t['audit_vocab_adopt']}**: " + ", ".join(item["vocabulary_to_adopt"])
                )

            if item.get("reference_hooks"):
                st.markdown(f"**{t['audit_ref_hooks']}**")
                for h in item["reference_hooks"]:
                    ex = f" — 例: 「{h['example']}」" if h.get("example") else ""
                    st.markdown(f"- `{h['pattern']}`{ex}")

            st.info(f"{t['caveat_prefix']}: {item['caveat']}")
            if item.get("needs_gemini_note"):
                st.caption(item["needs_gemini_note"])


with tab_insight:
    render_insight_tab(report)

with tab_audit:
    render_audit_tab()
