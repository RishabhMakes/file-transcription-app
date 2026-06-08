"""Core data types and backend dispatch for transcription."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Word:
    """A single word with timestamp."""

    text: str
    start: float
    end: float
    speaker: Optional[str] = None


@dataclass
class Segment:
    """A segment/sentence with timestamp."""

    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)
    speaker: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Complete transcription result with timestamps."""

    text: str
    duration: float
    segments: list[Segment]
    has_speakers: bool = False


class TranscriptionError(Exception):
    """Raised when transcription fails."""


def transcribe_audio(
    audio_path: Path,
    backend: str = "parakeet",
    model_name: Optional[str] = None,
    chunk_duration: float = 120.0,
    word_timestamps: bool = False,
    language: Optional[str] = None,
    diarize: bool = False,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    hf_token: Optional[str] = None,
    compute_type: str = "int8",
    batch_size: int = 8,
) -> TranscriptionResult:
    """Dispatch to the selected transcription backend.

    Backends:
        parakeet: NVIDIA Parakeet v3 via MLX. Fast on Apple Silicon. English +
            24 other European languages. No speaker diarization.
        whisperx: OpenAI Whisper (faster-whisper) + pyannote diarization.
            Multilingual including Hindi/Indic; supports speaker labels. Slower.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    backend = backend.lower()
    if backend == "parakeet":
        if diarize:
            raise TranscriptionError(
                "Diarization is not supported by the parakeet backend. "
                "Use --backend whisperx for speaker labels."
            )
        from .backends.parakeet import transcribe as _transcribe
        return _transcribe(
            audio_path=audio_path,
            model_name=model_name or "mlx-community/parakeet-tdt-0.6b-v3",
            chunk_duration=chunk_duration,
            word_timestamps=word_timestamps,
        )
    if backend == "whisperx":
        from .backends.whisperx import transcribe as _transcribe
        return _transcribe(
            audio_path=audio_path,
            model_name=model_name or "large-v3",
            language=language,
            word_timestamps=word_timestamps,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hf_token=hf_token,
            compute_type=compute_type,
            batch_size=batch_size,
        )
    raise TranscriptionError(
        f"Unknown backend: {backend!r}. Choose 'parakeet' or 'whisperx'."
    )
