"""
Local (offline, no-Gemini) video understanding.

video_audit.analyse_video_file() sends the whole clip to Gemini - accurate,
but 1 API call per video, and Gemini's free tier caps out fast (see the
"deep video" features in pipeline.py / video_audit.deepen_with_video_content).
This module extracts the same broad signals with free, local tools instead:

    speech      -> faster-whisper (runs on CPU, downloads a model once)
    on-screen   -> EasyOCR, sampled frames (pure Python, no system package)
    format/cuts -> PySceneDetect + OpenCV (shot-cut count, a mechanical proxy
                   for "static talking head" vs "fast-cut edit")

The trade-off is judgment, not just accuracy: Gemini can SAY "this is a POV
skit" or "this hook is weak". These tools can only report what they
literally detected (words spoken, text pixels, cut count) - detected_format
here is a coarse mechanical label, not a genre judgment. For hook/format
JUDGMENT, feed this module's output (spoken_hook, on_screen_text) into the
existing Gemini insight/audit prompts, which already take plain text and
cost only ONE batched call regardless of how many clips fed it.

Every capability degrades independently and silently (missing package,
model-download failure, unreadable video) - a clip that fails every step
still returns a Video with just caption/duration, never raises.
"""
from __future__ import annotations

from ingest import Video

_WHISPER_MODEL = None  # lazy singleton - loading it is the slow part
_OCR_READER = None     # lazy singleton - EasyOCR loads its net once


# --- Speech -> text (faster-whisper) -----------------------------------------

def _whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        from faster_whisper import WhisperModel
        _WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    return _WHISPER_MODEL


def _transcribe(path: str, hook_window_sec: float = 3.0) -> tuple[str, str]:
    """(full_transcript, spoken_hook) - spoken_hook is whatever was said in
    the first `hook_window_sec` seconds, same definition analyse_video_file()
    uses for Gemini. ("", "") on any failure (no package, bad audio, ...)."""
    try:
        model = _whisper_model()
        segments, _info = model.transcribe(path, vad_filter=True)
        full, hook = [], []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            full.append(text)
            if seg.start < hook_window_sec:
                hook.append(text)
        return " ".join(full).strip(), " ".join(hook).strip()
    except Exception:  # noqa: BLE001 - transcription is optional, never fatal
        return "", ""


# --- Frames, on-screen text (OpenCV + EasyOCR) -------------------------------

def _sample_frames(path: str, max_frames: int = 8):
    """Up to `max_frames` frames spread evenly across the clip, as numpy
    arrays (BGR, OpenCV's native order). [] on any decode failure."""
    try:
        import cv2
    except ImportError:
        return []
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return []
        step = max(1, total // max_frames)
        frames = []
        for idx in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
            if len(frames) >= max_frames:
                break
        return frames
    finally:
        cap.release()


def _ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr
        _OCR_READER = easyocr.Reader(["en", "ja"], gpu=False, verbose=False)
    return _OCR_READER


def _ocr_on_screen_text(frames) -> str:
    """Text overlays read off sampled frames, de-duplicated and joined with
    " / " (same join style analyse_video_file() uses). "" on any failure or
    if EasyOCR isn't installed."""
    if not frames:
        return ""
    try:
        reader = _ocr_reader()
    except Exception:  # noqa: BLE001 - OCR is optional, never fatal
        return ""
    seen: set[str] = set()
    lines: list[str] = []
    for frame in frames:
        try:
            for text in reader.readtext(frame, detail=0):
                cleaned = text.strip()
                key = cleaned.lower()
                if cleaned and len(cleaned) >= 2 and key not in seen:
                    seen.add(key)
                    lines.append(cleaned)
        except Exception:  # noqa: BLE001 - one bad frame shouldn't drop the rest
            continue
    return " / ".join(lines[:20])


# --- Cuts / mechanical format guess (PySceneDetect) --------------------------

def _count_cuts(path: str) -> int:
    """Number of hard shot cuts PySceneDetect finds. 0 on any failure (a
    genuinely static clip also legitimately scores 0, so this is never
    surfaced as an error - just folded into the format guess)."""
    try:
        from scenedetect import detect, ContentDetector
        scenes = detect(path, ContentDetector())
        return max(0, len(scenes) - 1)
    except Exception:  # noqa: BLE001 - format guess is optional, never fatal
        return 0


def _guess_format(cut_count: int, duration_sec: int, has_on_screen_text: bool) -> str:
    """A coarse MECHANICAL label from cut count alone - not a genre judgment
    the way Gemini's detected_format is. Good enough to tell "one continuous
    shot" from "fast-cut edit" apart; anything subtler needs an LLM."""
    if duration_sec <= 0:
        base = "unknown pacing"
    elif cut_count <= 1:
        base = "single continuous shot"
    elif cut_count >= max(3, duration_sec // 3):
        base = f"fast-cut edit ({cut_count} cuts)"
    else:
        base = f"multi-shot ({cut_count} cuts)"
    return base + (" + text overlay" if has_on_screen_text else "")


# --- Entry point --------------------------------------------------------------

def analyse_video_file_locally(
    path: str, caption: str = "", views: int = 0, video_id: str | None = None,
    url: str = "",
) -> Video:
    """Same signature/return shape as video_audit.analyse_video_file() - a
    drop-in, zero-Gemini-calls replacement. `caption` / `views` are passed
    through verbatim (same as the Gemini version); every other field is
    extracted locally: transcript + spoken_hook (faster-whisper), on_screen
    text (EasyOCR on sampled frames), duration + a mechanical detected_format
    (OpenCV + PySceneDetect). visual_tags stays empty - that needs semantic
    image understanding, which no tool here provides; only Gemini can fill it.

    Never raises: a clip where every local tool fails still returns a Video
    with just caption/duration/id, same "no video-content signal" outcome as
    analyse_video_file() failing would have - the caller doesn't need to
    special-case this path.
    """
    import os

    vid = video_id or os.path.splitext(os.path.basename(path))[0]

    duration_sec = 0
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if fps > 0:
                duration_sec = int(frame_count / fps)
        finally:
            cap.release()
    except Exception:  # noqa: BLE001 - duration is a nice-to-have
        pass

    transcript, spoken_hook = _transcribe(path)
    frames = _sample_frames(path)
    on_screen_text = _ocr_on_screen_text(frames)
    cut_count = _count_cuts(path)
    detected_format = _guess_format(cut_count, duration_sec, bool(on_screen_text))

    return Video(
        id=str(vid),
        url=url.strip(),
        caption=caption.strip(),
        creator="own_post",
        transcript=transcript,
        spoken_hook=spoken_hook,
        detected_format=detected_format,
        on_screen_text=on_screen_text,
        visual_tags=[],
        duration_sec=duration_sec,
        is_slideshow=False,
        engagement={"views": int(views)} if views else {},
    )
