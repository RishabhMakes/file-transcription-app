"""WhisperX backend (faster-whisper + optional pyannote diarization)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..transcribe import Segment, TranscriptionError, TranscriptionResult, Word


def _resolve_device() -> tuple[str, str]:
    """Return (device, default_compute_type) for whisperX.

    faster-whisper / CTranslate2 supports cuda and cpu only.
    """
    try:
        import torch  # noqa: F401

        if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _resolve_hf_token(hf_token: Optional[str]) -> Optional[str]:
    if hf_token:
        return hf_token
    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(env_var)
        if val:
            return val
    cache_path = Path.home() / ".cache" / "huggingface" / "token"
    if cache_path.exists():
        return cache_path.read_text().strip() or None
    return None


def transcribe(
    audio_path: Path,
    model_name: str = "large-v3",
    language: Optional[str] = None,
    word_timestamps: bool = False,
    diarize: bool = False,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    hf_token: Optional[str] = None,
    compute_type: str = "int8",
    batch_size: int = 8,
) -> TranscriptionResult:
    try:
        import whisperx
    except ImportError as e:
        raise TranscriptionError(
            "whisperx not installed. Install with: pip install '.[whisperx]'"
        ) from e

    device, default_compute = _resolve_device()
    if compute_type == "default":
        compute_type = default_compute

    try:
        audio = whisperx.load_audio(str(audio_path))

        model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        asr_result = model.transcribe(audio, batch_size=batch_size, language=language)

        detected_language = asr_result.get("language", language or "en")

        try:
            align_model, align_meta = whisperx.load_align_model(
                language_code=detected_language, device=device
            )
            aligned = whisperx.align(
                asr_result["segments"],
                align_model,
                align_meta,
                audio,
                device,
                return_char_alignments=False,
            )
            segments_data = aligned["segments"]
        except Exception:
            segments_data = asr_result["segments"]

        if diarize:
            token = _resolve_hf_token(hf_token)
            if not token:
                raise TranscriptionError(
                    "Diarization needs a HuggingFace token. Set HF_TOKEN env var, "
                    "run `hf auth login`, or pass --hf-token. You must also accept "
                    "the licenses for pyannote/segmentation-3.0, "
                    "pyannote/speaker-diarization-3.1, and "
                    "pyannote/speaker-diarization-community-1 on huggingface.co."
                )
            diarize_pipeline = whisperx.diarize.DiarizationPipeline(
                use_auth_token=token, device=device
            )
            diarize_segments = diarize_pipeline(
                audio, min_speakers=min_speakers, max_speakers=max_speakers
            )
            segments_data = whisperx.assign_word_speakers(
                diarize_segments, {"segments": segments_data}
            )["segments"]

        segments: list[Segment] = []
        max_end_time = 0.0
        full_text_parts: list[str] = []
        for seg in segments_data:
            words: list[Word] = []
            if word_timestamps:
                for w in seg.get("words", []) or []:
                    if "start" in w and "end" in w:
                        words.append(
                            Word(
                                text=str(w.get("word", "")).strip(),
                                start=float(w["start"]),
                                end=float(w["end"]),
                                speaker=w.get("speaker"),
                            )
                        )
            text = str(seg.get("text", "")).strip()
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            segments.append(
                Segment(
                    text=text,
                    start=start,
                    end=end,
                    words=words,
                    speaker=seg.get("speaker"),
                )
            )
            max_end_time = max(max_end_time, end)
            if text:
                full_text_parts.append(text)

        return TranscriptionResult(
            text=" ".join(full_text_parts),
            duration=max_end_time,
            segments=segments,
            has_speakers=diarize,
        )
    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(f"WhisperX transcription failed: {e}") from e
