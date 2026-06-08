"""Parakeet MLX backend (NVIDIA Parakeet v3 on Apple Silicon)."""

from __future__ import annotations

from pathlib import Path

from ..transcribe import Segment, TranscriptionError, TranscriptionResult, Word


def transcribe(
    audio_path: Path,
    model_name: str = "mlx-community/parakeet-tdt-0.6b-v3",
    chunk_duration: float = 120.0,
    word_timestamps: bool = False,
) -> TranscriptionResult:
    try:
        from parakeet_mlx import from_pretrained
    except ImportError as e:
        raise TranscriptionError(
            "parakeet-mlx not installed. Install with: pip install parakeet-mlx"
        ) from e

    try:
        model = from_pretrained(model_name)
        result = model.transcribe(
            str(audio_path),
            chunk_duration=chunk_duration,
            overlap_duration=15.0,
        )

        segments: list[Segment] = []
        max_end_time = 0.0
        for sentence in result.sentences:
            segment_words: list[Word] = []
            if word_timestamps and hasattr(result, "tokens"):
                for token in result.tokens:
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

        return TranscriptionResult(
            text=result.text.strip(),
            duration=max_end_time,
            segments=segments,
            has_speakers=False,
        )
    except Exception as e:
        raise TranscriptionError(f"Parakeet transcription failed: {e}") from e
