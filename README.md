# video-transcribe

CLI tool to turn videos into text using NVIDIA Parakeet v3 (MLX) models on Apple Silicon. It extracts audio with `ffmpeg`, runs transcription with `parakeet-mlx`, and writes subtitles/transcripts in multiple formats.

## Requirements
- Python 3.10+
- `ffmpeg` available on `PATH` (e.g., `brew install ffmpeg`)
- Apple Silicon with the MLX stack (for `parakeet-mlx` models)

## Installation
```bash
pip install .
# or editable for local work
pip install -e .
```

## Quick Start
```bash
video-transcribe path/to/video.mp4
# choose format
video-transcribe video.mp4 --format json
# custom output path
video-transcribe video.mp4 -o subtitles.srt
# word-level timestamps (srt/vtt/json)
video-transcribe video.mp4 --word-timestamps
```

## CLI Options
- `video_path` (positional): input video file.
- `-o, --output PATH`: output file path. Defaults to `<video_basename>.<format>`.
- `-f, --format [txt|srt|vtt|json]`: output format. Default `srt`.
- `--word-timestamps, -w`: include word-level timing (srt, vtt, json).
- `--chunk-duration FLOAT`: seconds per audio chunk. Default `120.0`.
- `-m, --model TEXT`: Parakeet model name. Default `mlx-community/parakeet-tdt-0.6b-v3`.
- `--keep-audio`: keep the extracted WAV instead of deleting it.

## Output Formats
- `txt`: plain text transcript.
- `srt`: subtitle cues per segment; with `--word-timestamps` each word becomes a cue.
- `vtt`: WebVTT; supports word-level cues when enabled.
- `json`: structured transcript `{text, duration, segments[, words]}` with optional word timings.

## Architecture (brief)
- `video_transcribe.cli`: Typer CLI wiring, argument parsing, progress display, and high-level flow (extract → transcribe → save).
- `video_transcribe.audio`: `ffmpeg` wrapper to extract 16 kHz mono WAV; checks tool availability and errors clearly.
- `video_transcribe.transcribe`: loads Parakeet MLX model, chunks audio, builds `TranscriptionResult` with segments and optional word timings.
- `video_transcribe.formats`: converters for `txt`, `srt`, `vtt`, `json` and writer function `save_transcript`.
- `__init__.py`: package version.

## Examples
```bash
# Basic
video-transcribe test_video.mp4

# Word-level JSON for downstream processing
video-transcribe test_video.mp4 --format json --word-timestamps -o transcript.json

# Longer videos with larger chunks and custom model
video-transcribe lecture.mp4 --chunk-duration 180 -m mlx-community/parakeet-tdt-1.7b-v3
```

## Notes
- If you see `ffmpeg not found`, install it and re-run.
- If `parakeet-mlx` is missing, install via `pip install parakeet-mlx`.
- Temporary audio is deleted unless `--keep-audio` is set.

