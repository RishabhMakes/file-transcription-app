"""CLI entry point for video-transcribe."""

from __future__ import annotations

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
    help="Transcribe video and audio files. Backends: Parakeet (fast, English/European) "
    "or WhisperX (multilingual + diarization).",
    add_completion=False,
)
console = Console()


class OutputFormat(str, Enum):
    txt = "txt"
    srt = "srt"
    vtt = "vtt"
    json = "json"


class Backend(str, Enum):
    parakeet = "parakeet"
    whisperx = "whisperx"


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
        typer.Option("-f", "--format", help="Output format."),
    ] = OutputFormat.srt,
    backend: Annotated[
        Backend,
        typer.Option(
            "-b",
            "--backend",
            help="ASR backend. parakeet = fast English/European, "
            "whisperx = multilingual + optional diarization.",
        ),
    ] = Backend.parakeet,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model name. Default: parakeet → mlx-community/parakeet-tdt-0.6b-v3, "
            "whisperx → large-v3.",
        ),
    ] = None,
    language: Annotated[
        Optional[str],
        typer.Option(
            "--language",
            "-l",
            help="Language code (whisperx only, e.g. 'en', 'hi'). Omit to auto-detect.",
        ),
    ] = None,
    diarize: Annotated[
        bool,
        typer.Option(
            "--diarize",
            help="Add speaker labels (whisperx only; requires HF token + accepted "
            "pyannote licenses).",
        ),
    ] = False,
    min_speakers: Annotated[
        Optional[int],
        typer.Option("--min-speakers", help="Minimum number of speakers (whisperx + diarize)."),
    ] = None,
    max_speakers: Annotated[
        Optional[int],
        typer.Option("--max-speakers", help="Maximum number of speakers (whisperx + diarize)."),
    ] = None,
    hf_token: Annotated[
        Optional[str],
        typer.Option(
            "--hf-token",
            help="HuggingFace token for diarization. Falls back to HF_TOKEN env var "
            "or ~/.cache/huggingface/token.",
        ),
    ] = None,
    compute_type: Annotated[
        str,
        typer.Option(
            "--compute-type",
            help="Compute type for whisperx (int8|float16|float32|default).",
        ),
    ] = "int8",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Batch size for whisperx inference."),
    ] = 8,
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
        typer.Option("--chunk-duration", help="Chunk duration for parakeet (seconds)."),
    ] = 120.0,
    keep_audio: Annotated[
        bool,
        typer.Option("--keep-audio", help="Keep the extracted/converted audio file."),
    ] = False,
) -> None:
    """Transcribe a video or audio file to text with timestamps.

    Examples:

        video-transcribe podcast.mp3

        video-transcribe call.m4a --backend whisperx --language hi --diarize \
            --min-speakers 2 --max-speakers 2

        video-transcribe video.mp4 --format json --word-timestamps
    """
    if output is None:
        output = input_path.with_suffix(f".{format.value}")

    audio_path: Optional[Path] = None

    try:
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

        backend_label = backend.value
        if backend == Backend.whisperx and diarize:
            backend_label += " + diarization"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Transcribing with {backend_label}...", total=None)
            result = transcribe_audio(
                audio_path,
                backend=backend.value,
                model_name=model,
                chunk_duration=chunk_duration,
                word_timestamps=word_timestamps,
                language=language,
                diarize=diarize,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                hf_token=hf_token,
                compute_type=compute_type,
                batch_size=batch_size,
            )

        speaker_note = " with speakers" if result.has_speakers else ""
        console.print(
            f"[green]✓[/green] Transcription complete "
            f"({result.duration:.1f}s audio{speaker_note})"
        )

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
        if audio_path and audio_path.exists() and not keep_audio:
            audio_path.unlink()


if __name__ == "__main__":
    app()
