# AV Sync Detector

Live A/V sync detector for comparing a source stream URL with an output stream
URL using content fingerprints instead of timestamps.

It is built for checking whether a transcoder, packager, or distribution server
has introduced audio/video drift. It does not record media to disk. It keeps
rolling audio/video fingerprints in memory and reports whether the source and
output are aligned.

## What It Measures

The detector estimates:

- source-to-output audio latency
- source-to-output video latency
- A/V offset, calculated as:

```text
av_offset_ms = (video_latency_s - audio_latency_s) * 1000
```

Large source-to-output latency is not a failure by itself. If audio and video
share the same delay, the stream is considered aligned.

## Requirements

- Python 3.11 or newer
- FFmpeg available in `PATH`

Install FFmpeg first:

### macOS

```bash
brew install ffmpeg
```

### Debian / Ubuntu

```bash
sudo apt update
sudo apt install -y ffmpeg python3 python3-venv
```

### Fedora

```bash
sudo dnf install -y ffmpeg python3
```

### Windows

Install Python 3.11+ from [python.org](https://www.python.org/downloads/windows/)
or the Microsoft Store. Install FFmpeg with one of:

```powershell
winget install Gyan.FFmpeg
```

or:

```powershell
choco install ffmpeg
```

After installing FFmpeg, open a new terminal and verify:

```bash
ffmpeg -version
python --version
```

On Windows, use `py -3 --version` if `python` is not mapped.

## Install

### From A Clone

macOS / Linux:

```bash
git clone <your-repo-url> av-sync-detector
cd av-sync-detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
git clone <your-repo-url> av-sync-detector
cd av-sync-detector
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Verify the install:

```bash
avsync-detector --help
pytest -q
```

### Directly From GitHub

After you publish the repo, users can install from GitHub:

```bash
python -m pip install "git+https://github.com/<owner>/av-sync-detector.git"
```

For isolated command-line installs, `pipx` also works:

```bash
pipx install "git+https://github.com/<owner>/av-sync-detector.git"
```

## Usage

### JSON Mode

Use JSON mode for scripts, automation, simple terminals, and logs:

```bash
avsync-detector json \
  --source "https://example.com/source/index.m3u8" \
  --output "https://example.com/output/index.m3u8" \
  --duration 70
```

JSON mode prints one object and exits. By default it exits with code `0` after
printing the result. Add `--fail-on-bad` if automation should return nonzero
for `out_of_sync` or `inconclusive`.

### TUI Mode

Use TUI mode for live interactive checks:

```bash
avsync-detector tui \
  --source "https://example.com/source/index.m3u8" \
  --output "https://example.com/output/index.m3u8"
```

The TUI shows:

- A/V offset and direction
- source-to-output audio/video latency
- confidence
- decode health
- rolling offset history

Press `Ctrl+C` to exit.

## Important Runtime Notes

Let the detector run long enough for the output distribution delay. HLS
distribution can easily be 30 seconds or more behind the source. The default
search window is 60 seconds and the rolling RAM window is 120 seconds.

For high-latency outputs, prefer:

```bash
avsync-detector json \
  --source "<source-url>" \
  --output "<output-url>" \
  --duration 70 \
  --window 120 \
  --max-latency 60
```

If the detector reports `inconclusive` with warnings like
`min_latency_boundary`, `max_latency_boundary`, or `low_confidence`, do not
treat the visible latency guesses as a sync verdict. Run longer or widen the
search window.

## Options

Common options:

```text
--source URL          source stream URL
--output URL          output stream URL
--duration SECONDS    JSON mode observation time
--window SECONDS      rolling RAM window, default 120
--max-latency SECONDS maximum source-to-output latency search, default 60
--min-overlap SECONDS minimum matched content overlap, default 6
--video-rate FPS      video fingerprint sample rate, default 20
--ok-ms MS            aligned threshold, default 120
--warn-ms MS          warning threshold, default 250
```

## How It Works

For each URL, the detector runs one FFmpeg process and extracts both audio and
video from that same input session. This avoids false A/V offsets caused by
joining live audio and video at different positions.

Audio is converted into compact energy fingerprints. Video is converted into
localized block-motion fingerprints so mostly static talking-head shots can
still be matched when only the mouth or face moves.

See [docs/algorithm.md](docs/algorithm.md) for details and limitations.

## Development

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Project layout:

```text
src/avsync_detector/   package source
tests/                 unit tests
docs/                  user and algorithm documentation
.github/workflows/     GitHub Actions CI
```

## License

No license has been selected yet. Add a license before publishing if you want
other people to reuse, modify, or redistribute the code.
