"""Transcription using Parakeet MLX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Word:
    """A single word with timestamp."""

    text: str
    start: float
    end: float


@dataclass
class Segment:
    """A segment/sentence with timestamp."""

    text: str
    start: float
    end: float
    words: list[Word]


@dataclass
class TranscriptionResult:
    """Complete transcription result with timestamps."""

    text: str
    duration: float
    segments: list[Segment]


class TranscriptionError(Exception):
    """Raised when transcription fails."""

    pass


def transcribe_audio(
    audio_path: Path,
    model_name: str = "mlx-community/parakeet-tdt-0.6b-v3",
    chunk_duration: float = 120.0,
    word_timestamps: bool = False,
) -> TranscriptionResult:
    """
    Transcribe an audio file using Parakeet MLX.

    Args:
        audio_path: Path to the audio file (WAV format, 16kHz mono).
        model_name: Name of the Parakeet model to use.
        chunk_duration: Duration in seconds for chunking long audio.
        word_timestamps: Whether to include word-level timestamps.

    Returns:
        TranscriptionResult with text, duration, and segments.

    Raises:
        TranscriptionError: If transcription fails.
    """
    try:
        from parakeet_mlx import from_pretrained
    except ImportError:
        raise TranscriptionError(
            "parakeet-mlx not installed. Please install it with: pip install parakeet-mlx"
        )

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        # Load the model
        model = from_pretrained(model_name)

        # Transcribe with chunking for long audio
        result = model.transcribe(
            str(audio_path),
            chunk_duration=chunk_duration,
            overlap_duration=15.0,
        )

        # Build segments from sentences
        segments = []
        max_end_time = 0.0
        for sentence in result.sentences:
            # Collect words for this segment if word timestamps requested
            segment_words = []
            if word_timestamps and hasattr(result, "tokens"):
                for token in result.tokens:
                    # Check if token falls within this sentence's time range
                    if (
                        hasattr(token, "start")
                        and hasattr(token, "end")
                        and token.start >= sentence.start
                        and token.end <= sentence.end
                    ):
                        segment_words.append(
                            Word(
                                text=token.text.strip(),
                                start=token.start,
                                end=token.end,
                            )
                        )

            segments.append(
                Segment(
                    text=sentence.text.strip(),
                    start=sentence.start,
                    end=sentence.end,
                    words=segment_words,
                )
            )
            max_end_time = max(max_end_time, sentence.end)

        # Calculate duration from the last segment's end time
        duration = max_end_time

        return TranscriptionResult(
            text=result.text.strip(),
            duration=duration,
            segments=segments,
        )

    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}")
