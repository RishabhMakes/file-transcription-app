"""Audio extraction from video and audio files using ffmpeg."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif"}


class AudioExtractionError(Exception):
    """Raised when audio extraction fails."""

    pass


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def is_audio_file(path: Path) -> bool:
    """Return True if the file extension indicates an audio-only file."""
    return path.suffix.lower() in AUDIO_EXTENSIONS


def extract_audio(input_path: Path, output_path: Path | None = None) -> Path:
    """
    Convert a video or audio file to 16kHz mono WAV for transcription.

    Args:
        input_path: Path to the input video or audio file.
        output_path: Optional path for the output WAV file.
                    If None, creates a temporary file.

    Returns:
        Path to the extracted/converted audio file.

    Raises:
        AudioExtractionError: If extraction fails.
        FileNotFoundError: If the input file doesn't exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not check_ffmpeg():
        raise AudioExtractionError(
            "ffmpeg not found. Please install it with: brew install ffmpeg"
        )

    # Create output path if not provided
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = Path(temp_dir) / f"{input_path.stem}_audio.wav"

    # ffmpeg command to convert to 16kHz mono WAV
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vn",                    # Ignore video stream (no-op for audio files)
        "-acodec", "pcm_s16le",   # PCM 16-bit little-endian
        "-ar", "16000",           # 16kHz sample rate
        "-ac", "1",               # Mono
        "-y",                     # Overwrite output
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else "Unknown error"
        raise AudioExtractionError(f"Failed to process audio: {stderr}")

    if not output_path.exists():
        raise AudioExtractionError("Audio processing failed: output file not created")

    return output_path
