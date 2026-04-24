# video-transcribe

CLI tool to turn videos and audio files into text using NVIDIA Parakeet v3 (MLX) models on Apple Silicon. It processes audio with `ffmpeg`, runs transcription with `parakeet-mlx`, and writes subtitles/transcripts in multiple formats.

Supported input formats:
- **Video**: MP4, MOV, AVI, MKV, WebM, and anything else `ffmpeg` can read
- **Audio**: MP3, WAV, M4A, AAC, FLAC, OGG, Opus, WMA, AIFF

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
# audio files work too
video-transcribe podcast.mp3
video-transcribe interview.m4a
# choose format
video-transcribe video.mp4 --format json
# custom output path
video-transcribe audio.mp3 -o subtitles.srt
# word-level timestamps (srt/vtt/json)
video-transcribe video.mp4 --word-timestamps
```

## CLI Options
- `input_path` (positional): input video or audio file.
- `-o, --output PATH`: output file path. Defaults to `<input_basename>.<format>`.
- `-f, --format [txt|srt|vtt|json]`: output format. Default `srt`.
- `--word-timestamps, -w`: include word-level timing (srt, vtt, json).
- `--chunk-duration FLOAT`: seconds per audio chunk. Default `120.0`.
- `-m, --model TEXT`: Parakeet model name. Default `mlx-community/parakeet-tdt-0.6b-v3`.
- `--keep-audio`: keep the extracted/converted WAV instead of deleting it.

## Output Formats
- `txt`: plain text transcript.
- `srt`: subtitle cues per segment; with `--word-timestamps` each word becomes a cue.
- `vtt`: WebVTT; supports word-level cues when enabled.
- `json`: structured transcript `{text, duration, segments[, words]}` with optional word timings.

## Architecture (brief)
- `video_transcribe.cli`: Typer CLI wiring, argument parsing, progress display, and high-level flow (extract/convert → transcribe → save).
- `video_transcribe.audio`: `ffmpeg` wrapper to convert any video or audio input to 16 kHz mono WAV; checks tool availability and errors clearly.
- `video_transcribe.transcribe`: loads Parakeet MLX model, chunks audio, builds `TranscriptionResult` with segments and optional word timings.
- `video_transcribe.formats`: converters for `txt`, `srt`, `vtt`, `json` and writer function `save_transcript`.
- `__init__.py`: package version.

## Examples
```bash
# Basic video
video-transcribe test_video.mp4

# Audio file
video-transcribe podcast.mp3

# Word-level JSON for downstream processing
video-transcribe interview.m4a --format json --word-timestamps -o transcript.json

# Longer files with larger chunks and custom model
video-transcribe lecture.mp4 --chunk-duration 180 -m mlx-community/parakeet-tdt-1.7b-v3
```

## Notes
- If you see `ffmpeg not found`, install it and re-run.
- If `parakeet-mlx` is missing, install via `pip install parakeet-mlx`.
- Temporary audio is deleted unless `--keep-audio` is set.

