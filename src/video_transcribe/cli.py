"""CLI entry point for video-transcribe."""

from __future__ import annotations

import tempfile
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .audio import AudioExtractionError, extract_audio, is_audio_file
from .formats import save_transcript
from .transcribe import TranscriptionError, transcribe_audio

app = typer.Typer(
    name="video-transcribe",
    help="Transcribe video and audio files to text using Parakeet MLX (Apple Silicon).",
    add_completion=False,
)
console = Console()


class OutputFormat(str, Enum):
    """Supported output formats."""

    txt = "txt"
    srt = "srt"
    vtt = "vtt"
    json = "json"


@app.command()
def main(
    input_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the video or audio file to transcribe.",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            help="Output file path. Defaults to input filename with appropriate extension.",
        ),
    ] = None,
    format: Annotated[
        OutputFormat,
        typer.Option(
            "-f",
            "--format",
            help="Output format.",
        ),
    ] = OutputFormat.srt,
    word_timestamps: Annotated[
        bool,
        typer.Option(
            "--word-timestamps",
            "-w",
            help="Include word-level timestamps (for srt, vtt, json).",
        ),
    ] = False,
    chunk_duration: Annotated[
        float,
        typer.Option(
            "--chunk-duration",
            help="Duration in seconds for chunking long audio.",
        ),
    ] = 120.0,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Parakeet model to use.",
        ),
    ] = "mlx-community/parakeet-tdt-0.6b-v3",
    keep_audio: Annotated[
        bool,
        typer.Option(
            "--keep-audio",
            help="Keep the extracted/converted audio file.",
        ),
    ] = False,
) -> None:
    """
    Transcribe a video or audio file to text with timestamps.

    Examples:

        video-transcribe video.mp4

        video-transcribe audio.mp3 --format json --word-timestamps

        video-transcribe podcast.m4a -o transcript.srt
    """
    # Determine output path
    if output is None:
        output = input_path.with_suffix(f".{format.value}")

    audio_path: Optional[Path] = None

    try:
        # Step 1: Extract/convert audio
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if is_audio_file(input_path):
                progress.add_task("Converting audio to required format...", total=None)
            else:
                progress.add_task("Extracting audio from video...", total=None)
            audio_path = extract_audio(input_path)

        console.print(f"[green]✓[/green] Audio ready: {audio_path}")

        # Step 2: Transcribe
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(
                f"Transcribing with {model.split('/')[-1]}...", total=None
            )
            result = transcribe_audio(
                audio_path,
                model_name=model,
                chunk_duration=chunk_duration,
                word_timestamps=word_timestamps,
            )

        console.print(f"[green]✓[/green] Transcription complete ({result.duration:.1f}s audio)")

        # Step 3: Save output
        save_transcript(result, output, format.value, word_timestamps)
        console.print(f"[green]✓[/green] Saved to: {output}")

    except AudioExtractionError as e:
        console.print(f"[red]Error extracting audio:[/red] {e}")
        raise typer.Exit(1)
    except TranscriptionError as e:
        console.print(f"[red]Error during transcription:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        # Cleanup temporary audio file
        if audio_path and audio_path.exists() and not keep_audio:
            audio_path.unlink()


if __name__ == "__main__":
    app()
