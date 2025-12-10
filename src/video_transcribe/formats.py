"""Output formatters for transcription results."""

from __future__ import annotations

import json
from pathlib import Path

from .transcribe import TranscriptionResult


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def to_txt(result: TranscriptionResult) -> str:
    """Convert transcription to plain text."""
    return result.text


def to_srt(result: TranscriptionResult, word_timestamps: bool = False) -> str:
    """
    Convert transcription to SRT subtitle format.

    Args:
        result: Transcription result.
        word_timestamps: If True, create entries for each word.

    Returns:
        SRT formatted string.
    """
    lines = []
    counter = 1

    if word_timestamps:
        # Word-level SRT
        for segment in result.segments:
            for word in segment.words:
                if word.text.strip():
                    start = format_timestamp_srt(word.start)
                    end = format_timestamp_srt(word.end)
                    lines.append(f"{counter}")
                    lines.append(f"{start} --> {end}")
                    lines.append(word.text.strip())
                    lines.append("")
                    counter += 1
    else:
        # Segment-level SRT
        for segment in result.segments:
            start = format_timestamp_srt(segment.start)
            end = format_timestamp_srt(segment.end)
            lines.append(f"{counter}")
            lines.append(f"{start} --> {end}")
            lines.append(segment.text)
            lines.append("")
            counter += 1

    return "\n".join(lines)


def to_vtt(result: TranscriptionResult, word_timestamps: bool = False) -> str:
    """
    Convert transcription to WebVTT subtitle format.

    Args:
        result: Transcription result.
        word_timestamps: If True, create entries for each word.

    Returns:
        VTT formatted string.
    """
    lines = ["WEBVTT", ""]

    if word_timestamps:
        # Word-level VTT
        for segment in result.segments:
            for word in segment.words:
                if word.text.strip():
                    start = format_timestamp_vtt(word.start)
                    end = format_timestamp_vtt(word.end)
                    lines.append(f"{start} --> {end}")
                    lines.append(word.text.strip())
                    lines.append("")
    else:
        # Segment-level VTT
        for segment in result.segments:
            start = format_timestamp_vtt(segment.start)
            end = format_timestamp_vtt(segment.end)
            lines.append(f"{start} --> {end}")
            lines.append(segment.text)
            lines.append("")

    return "\n".join(lines)


def to_json(result: TranscriptionResult, word_timestamps: bool = False) -> str:
    """
    Convert transcription to JSON format.

    Args:
        result: Transcription result.
        word_timestamps: If True, include word-level timestamps.

    Returns:
        JSON formatted string.
    """
    data = {
        "text": result.text,
        "duration": result.duration,
        "segments": [],
    }

    for segment in result.segments:
        seg_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }

        if word_timestamps and segment.words:
            seg_data["words"] = [
                {
                    "word": word.text,
                    "start": word.start,
                    "end": word.end,
                }
                for word in segment.words
            ]

        data["segments"].append(seg_data)

    return json.dumps(data, indent=2, ensure_ascii=False)


def save_transcript(
    result: TranscriptionResult,
    output_path: Path,
    format: str,
    word_timestamps: bool = False,
) -> None:
    """
    Save transcription result to a file.

    Args:
        result: Transcription result.
        output_path: Path to save the output file.
        format: Output format (txt, srt, vtt, json).
        word_timestamps: Include word-level timestamps.
    """
    formatters = {
        "txt": lambda r: to_txt(r),
        "srt": lambda r: to_srt(r, word_timestamps),
        "vtt": lambda r: to_vtt(r, word_timestamps),
        "json": lambda r: to_json(r, word_timestamps),
    }

    if format not in formatters:
        raise ValueError(f"Unknown format: {format}. Supported: {list(formatters.keys())}")

    content = formatters[format](result)
    output_path.write_text(content, encoding="utf-8")
