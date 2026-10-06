"""
Transcription layer (Whisper) - STUB for Stage 1a.

In Stage 1a our sample data already contains a `transcript` field, so we do
NOT run Whisper. This module exists so that when real video/audio arrives,
you drop the implementation in here and call `transcribe_file()` from the
ingestion step - without touching the rest of the pipeline.

To actually use it later:
    pip install faster-whisper
and implement the body below. Everything downstream already expects a plain
string transcript, so nothing else changes.
"""
from __future__ import annotations


def transcribe_file(audio_or_video_path: str, model_size: str = "base") -> str:
    """Return a transcript for a local audio/video file.

    Stage 1a: not used. Left intentionally unimplemented.

    Reference implementation for later (local, free, runs on CPU):

        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(audio_or_video_path)
        return " ".join(seg.text for seg in segments).strip()
    """
    raise NotImplementedError(
        "Transcription is not used in Stage 1a (sample data ships with "
        "transcripts). Implement this when real video arrives."
    )
