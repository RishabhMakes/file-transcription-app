# video-transcribe

CLI tool to turn videos and audio files into text. Two backends, pick what you need in the moment:

- **Parakeet** (default) — NVIDIA Parakeet v3 via MLX. Fast on Apple Silicon. English + 24 European languages. No speaker diarization.
- **WhisperX** — OpenAI Whisper (faster-whisper) + pyannote diarization. Multilingual including Hindi/Indic. Supports speaker labels. Slower, especially on CPU.

Audio is processed through `ffmpeg` (16 kHz mono WAV) then handed to the chosen backend. Output as `txt`, `srt`, `vtt`, or `json`.

## When to use which

| You want… | Use |
|---|---|
| Fast English/European transcription on Apple Silicon | `--backend parakeet` (default) |
| Hindi / Indic / non-European audio | `--backend whisperx --language hi` |
| Speaker labels ("who said what") | `--backend whisperx --diarize` |
| Code-switched Hinglish / multilingual call | `--backend whisperx` (omit `--language` to auto-detect) |
| Lowest-latency, no setup beyond ffmpeg | Parakeet |

## Requirements

- Python 3.10+
- `ffmpeg` on `PATH` (`brew install ffmpeg`)
- Apple Silicon recommended for Parakeet (MLX). WhisperX runs on CPU or CUDA — MPS not supported by faster-whisper.

## Installation

Base install (Parakeet only):
```bash
pip install .
```

With WhisperX support (heavier — pulls torch, ctranslate2, pyannote):
```bash
pip install ".[whisperx]"
```

For diarization (whisperx + `--diarize`) you also need:
1. A free HuggingFace account
2. Accept the licenses on these three model pages:
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/speaker-diarization-community-1
3. A read-only HF token (`hf auth login` or `export HF_TOKEN=hf_...`)

The token only needs the fine-grained scope **"Read access to contents of all public gated repos you can access"**.

## Quick Start

```bash
# Fast path: English, no speakers
video-transcribe podcast.mp3

# Hinglish phone call with speaker labels
video-transcribe call.m4a --backend whisperx --diarize --min-speakers 2 --max-speakers 2

# Hindi audio
video-transcribe interview.mp4 --backend whisperx --language hi

# Word-level JSON for downstream processing
video-transcribe lecture.mp3 --format json --word-timestamps

# Custom output path
video-transcribe audio.mp3 -o subtitles.srt
```

## CLI Options

- `input_path` (positional): input video or audio file.
- `-o, --output PATH`: output file path. Defaults to `<input_basename>.<format>`.
- `-f, --format [txt|srt|vtt|json]`: output format. Default `srt`.
- `-b, --backend [parakeet|whisperx]`: ASR backend. Default `parakeet`.
- `-m, --model TEXT`: model name. Default: `mlx-community/parakeet-tdt-0.6b-v3` (parakeet) or `large-v3` (whisperx).
- `-l, --language TEXT`: language code for whisperx (`en`, `hi`, etc.). Omit to auto-detect.
- `--diarize`: add speaker labels (whisperx only).
- `--min-speakers INT` / `--max-speakers INT`: bounds for diarization clustering.
- `--hf-token TEXT`: HuggingFace token for diarization. Falls back to `HF_TOKEN` env var or `~/.cache/huggingface/token`.
- `--compute-type TEXT`: whisperx compute type (`int8` default for CPU, `float16` for CUDA).
- `--batch-size INT`: whisperx batch size. Default `8`.
- `--word-timestamps, -w`: include word-level timing (srt, vtt, json).
- `--chunk-duration FLOAT`: seconds per audio chunk for parakeet. Default `120.0`.
- `--keep-audio`: keep the extracted WAV instead of deleting it.

## Output Formats

- `txt`: plain text transcript. With diarization, one segment per line prefixed with `[SPEAKER_XX]: `.
- `srt`: subtitle cues per segment. Speaker prefix included when diarized. With `--word-timestamps` each word becomes a cue.
- `vtt`: WebVTT; speaker prefix and word-level cues like SRT.
- `json`: structured `{text, duration, has_speakers, segments[, words]}`. Speaker labels surface as `speaker` fields.

## Architecture

- `video_transcribe.cli`: Typer CLI — argument parsing, progress display, flow.
- `video_transcribe.audio`: `ffmpeg` wrapper to convert any input to 16 kHz mono WAV.
- `video_transcribe.transcribe`: dataclasses (`Word`, `Segment`, `TranscriptionResult`) + backend dispatch.
- `video_transcribe.backends.parakeet`: Parakeet MLX implementation.
- `video_transcribe.backends.whisperx`: WhisperX + optional pyannote diarization.
- `video_transcribe.formats`: converters for `txt`, `srt`, `vtt`, `json` (speaker-aware).

## Notes

- If you see `ffmpeg not found`, install it and re-run.
- WhisperX on Apple Silicon runs on CPU (no MPS support in CTranslate2). A 60-min audio file takes roughly 15–30 min with `large-v3`, plus 20–40 min for diarization.
- The default `large-v3` Whisper model is ~1.5 GB and downloads on first use.
- Parakeet v3 ≠ Hindi support. v3 added 24 European languages; Hindi and other Indic languages are not covered. Use WhisperX for those.
- Temporary audio is deleted unless `--keep-audio` is set.
