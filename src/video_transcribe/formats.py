"""Output formatters for transcription results."""

from __future__ import annotations

import json
from pathlib import Path

from .transcribe import Segment, TranscriptionResult


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


def _line(segment: Segment, text: str) -> str:
    if segment.speaker:
        return f"[{segment.speaker}]: {text}"
    return text


def to_txt(result: TranscriptionResult) -> str:
    """Plain text. If diarized, one segment per line with speaker prefix."""
    if result.has_speakers:
        return "\n".join(_line(s, s.text) for s in result.segments if s.text)
    return result.text


def to_srt(result: TranscriptionResult, word_timestamps: bool = False) -> str:
    lines: list[str] = []
    counter = 1

    if word_timestamps:
        for segment in result.segments:
            for word in segment.words:
                if word.text.strip():
                    start = format_timestamp_srt(word.start)
                    end = format_timestamp_srt(word.end)
                    speaker = word.speaker or segment.speaker
                    text = f"[{speaker}]: {word.text.strip()}" if speaker else word.text.strip()
                    lines.append(f"{counter}")
                    lines.append(f"{start} --> {end}")
                    lines.append(text)
                    lines.append("")
                    counter += 1
    else:
        for segment in result.segments:
            start = format_timestamp_srt(segment.start)
            end = format_timestamp_srt(segment.end)
            lines.append(f"{counter}")
            lines.append(f"{start} --> {end}")
            lines.append(_line(segment, segment.text))
            lines.append("")
            counter += 1

    return "\n".join(lines)


def to_vtt(result: TranscriptionResult, word_timestamps: bool = False) -> str:
    lines = ["WEBVTT", ""]

    if word_timestamps:
        for segment in result.segments:
            for word in segment.words:
                if word.text.strip():
                    start = format_timestamp_vtt(word.start)
                    end = format_timestamp_vtt(word.end)
                    speaker = word.speaker or segment.speaker
                    text = f"[{speaker}]: {word.text.strip()}" if speaker else word.text.strip()
                    lines.append(f"{start} --> {end}")
                    lines.append(text)
                    lines.append("")
    else:
        for segment in result.segments:
            start = format_timestamp_vtt(segment.start)
            end = format_timestamp_vtt(segment.end)
            lines.append(f"{start} --> {end}")
            lines.append(_line(segment, segment.text))
            lines.append("")

    return "\n".join(lines)


def to_json(result: TranscriptionResult, word_timestamps: bool = False) -> str:
    data: dict = {
        "text": result.text,
        "duration": result.duration,
        "has_speakers": result.has_speakers,
        "segments": [],
    }

    for segment in result.segments:
        seg_data: dict = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        if segment.speaker:
            seg_data["speaker"] = segment.speaker

        if word_timestamps and segment.words:
            seg_data["words"] = [
                {
                    "word": word.text,
                    "start": word.start,
                    "end": word.end,
                    **({"speaker": word.speaker} if word.speaker else {}),
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
