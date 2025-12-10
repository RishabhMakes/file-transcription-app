"""Audio extraction from video files using ffmpeg."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


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


def extract_audio(video_path: Path, output_path: Path | None = None) -> Path:
    """
    Extract audio from a video file as 16kHz mono WAV.

    Args:
        video_path: Path to the input video file.
        output_path: Optional path for the output WAV file.
                    If None, creates a temporary file.

    Returns:
        Path to the extracted audio file.

    Raises:
        AudioExtractionError: If extraction fails.
        FileNotFoundError: If the video file doesn't exist.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not check_ffmpeg():
        raise AudioExtractionError(
            "ffmpeg not found. Please install it with: brew install ffmpeg"
        )

    # Create output path if not provided
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = Path(temp_dir) / f"{video_path.stem}_audio.wav"

    # ffmpeg command to extract audio as 16kHz mono WAV
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",                    # No video
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
        raise AudioExtractionError(f"Failed to extract audio: {stderr}")

    if not output_path.exists():
        raise AudioExtractionError("Audio extraction failed: output file not created")

    return output_path
