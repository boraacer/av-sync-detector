# AV Sync Detector Agent Guide

AV Sync Detector is a standalone Python CLI/TUI tool for comparing a source stream URL with an output stream URL using content fingerprints instead of timestamps.

## Working Rules

- Keep this repository standalone. Do not depend on StreamVector internals, services, controller APIs, or local deployment scripts.
- Prefer small, targeted changes with tests. The detector is used to validate live stream sync, so false confidence is worse than an inconclusive result.
- Do not print, commit, or hard-code private stream URLs, tokens, credentials, or customer data.
- Keep generated files out of git: `.pytest_cache/`, `__pycache__/`, `.venv/`, `*.egg-info/`, build outputs, and OS metadata.
- Preserve cross-platform behavior for macOS, Linux, and Windows when changing CLI, install, or subprocess code.

## Project Layout

- `src/avsync_detector/` - package source.
- `tests/` - pytest test suite.
- `docs/` - user-facing and algorithm documentation when present.
- `.github/workflows/` - CI when present.
- `pyproject.toml` - package metadata, dependencies, and pytest config.

## Runtime Requirements

- Python 3.11 or newer.
- FFmpeg in `PATH`.
- The CLI must work through the `avsync-detector` console script and `python -m avsync_detector`.

## Key Commands

Install for development:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

Run JSON mode:

```bash
avsync-detector json --source "<source-url>" --output "<output-url>" --duration 70
```

Run TUI mode:

```bash
avsync-detector tui --source "<source-url>" --output "<output-url>"
```

## Algorithm Notes

- One FFmpeg input session per URL must extract both audio and video. Do not split one URL into independent audio and video ffmpeg sessions; that can create fake A/V offsets on live streams.
- Matches at search-window boundaries are not trustworthy and should remain inconclusive unless a future algorithm proves otherwise.
- Low-confidence results should not publish a concrete A/V offset. Preserve diagnostic latency values, but keep direction and offset unknown.
- Favor deterministic tests for adversarial cases: repeated content, static talking heads, small mouth motion, boundary matches, and long HLS distribution delay.

## Testing Guidance

- Add or update tests before changing detector behavior.
- Use focused tests first, then run the full suite.
- For live stream checks, do not include private URLs or tokens in committed files or test fixtures.
- A passing live check is useful evidence, but unit tests must cover the behavior before committing an algorithm change.

