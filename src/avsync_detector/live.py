from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, replace
import os
import re

import numpy as np

from .align import estimate_alignment
from .result import AlignmentResult, classify_offset, dedupe_warnings


@dataclass(frozen=True)
class LiveOptions:
    audio_sample_rate: int = 8000
    audio_feature_rate: int = 100
    video_rate: int = 20
    video_width: int = 96
    video_height: int = 54
    min_latency_s: float = 0.0
    max_latency_s: float = 60.0
    min_overlap_s: float = 6.0
    rolling_window_s: float = 120.0
    ok_ms: float = 120.0
    warn_ms: float = 250.0
    offset_stability_required: bool = False
    offset_stability_window: int = 5
    offset_stability_min_estimates: int = 3
    offset_stability_tolerance_ms: float = 120.0
    offset_hold_unknown_estimates: int = 5


@dataclass(frozen=True)
class PipeHealth:
    name: str
    running: bool
    samples: int
    last_update_age_s: float | None
    restarts: int = 0
    return_code: int | None = None
    error_tail: str = ""
    total_samples: int = 0


class Rolling1D:
    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self._data = np.empty(0, dtype=np.float32)
        self.total_samples = 0
        self._lock = threading.Lock()

    def append(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        with self._lock:
            self.total_samples += values.size
            self._data = np.concatenate([self._data, values.astype(np.float32, copy=False)])
            if self._data.size > self.max_samples:
                self._data = self._data[-self.max_samples :]

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return self._data.copy()

    def clear(self) -> None:
        with self._lock:
            self._data = np.empty(0, dtype=np.float32)


class RollingVectors:
    def __init__(self, max_rows: int, dims: int):
        self.max_rows = max_rows
        self.dims = dims
        self._data = np.empty((0, dims), dtype=np.float32)
        self.total_rows = 0
        self._lock = threading.Lock()

    def append(self, value: np.ndarray) -> None:
        row = np.asarray(value, dtype=np.float32).reshape(1, self.dims)
        with self._lock:
            self.total_rows += 1
            self._data = np.vstack([self._data, row])
            if self._data.shape[0] > self.max_rows:
                self._data = self._data[-self.max_rows :, :]

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return self._data.copy()

    def clear(self) -> None:
        with self._lock:
            self._data = np.empty((0, self.dims), dtype=np.float32)


class AVFeaturePipe:
    def __init__(self, name: str, url: str, options: LiveOptions):
        self.name = name
        self.url = url
        self.options = options
        self.audio_buffer = Rolling1D(int(options.rolling_window_s * options.audio_feature_rate))
        self.frame_size = options.video_width * options.video_height
        self.video_buffer = RollingVectors(int(options.rolling_window_s * options.video_rate), video_feature_dimensions())
        self.proc: subprocess.Popen[bytes] | None = None
        self.thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.audio_last_update: float | None = None
        self.video_last_update: float | None = None
        self._previous_video: np.ndarray | None = None
        self.restarts = 0
        self.return_code: int | None = None
        self.error_tail = ""

    def start(self) -> None:
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._supervise, daemon=True)
        self.thread.start()

    def _supervise(self) -> None:
        while not self._stop_event.is_set():
            audio_r, audio_w = os.pipe()
            video_r, video_w = os.pipe()
            self._previous_video = None
            cmd = build_av_ffmpeg_cmd(self.url, self.options, audio_fd=audio_w, video_fd=video_w)
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    pass_fds=(audio_w, video_w),
                    close_fds=True,
                )
            finally:
                os.close(audio_w)
                os.close(video_w)
            with os.fdopen(audio_r, "rb", buffering=0) as audio_stream, os.fdopen(video_r, "rb", buffering=0) as video_stream:
                stderr_thread = threading.Thread(target=self._stderr_loop, args=(self.proc,), daemon=True)
                audio_thread = threading.Thread(target=self._read_audio_loop, args=(audio_stream,), daemon=True)
                video_thread = threading.Thread(target=self._read_video_loop, args=(video_stream,), daemon=True)
                stderr_thread.start()
                audio_thread.start()
                video_thread.start()
                self.return_code = self.proc.wait() if self.proc is not None else None
                audio_thread.join(timeout=1.0)
                video_thread.join(timeout=1.0)
            if self._stop_event.is_set():
                break
            self.restarts += 1
            time.sleep(2.0)

    def _read_audio_loop(self, stream) -> None:
        samples_per_feature = self.options.audio_sample_rate // self.options.audio_feature_rate
        chunk_bytes = max(samples_per_feature * 4, self.options.audio_sample_rate * 4 // 5)
        leftover = np.empty(0, dtype=np.float32)
        while not self._stop_event.is_set():
            raw = stream.read(chunk_bytes)
            if not raw:
                break
            values = np.frombuffer(raw, dtype=np.float32).copy()
            if leftover.size:
                values = np.concatenate([leftover, values])
            n = (values.size // samples_per_feature) * samples_per_feature
            if n:
                self.audio_buffer.append(audio_samples_to_features(values[:n], samples_per_feature=samples_per_feature))
                self.audio_last_update = time.time()
            leftover = values[n:]

    def _read_video_loop(self, stream) -> None:
        while not self._stop_event.is_set():
            raw = stream.read(self.frame_size)
            if len(raw) < self.frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            feature, self._previous_video = video_frame_to_feature(
                frame,
                previous=self._previous_video,
                width=self.options.video_width,
                height=self.options.video_height,
            )
            self.video_buffer.append(feature)
            self.video_last_update = time.time()

    def stop(self) -> None:
        self._stop_event.set()
        _stop_process(self.proc)

    def clear_current_window(self) -> None:
        self.audio_buffer.clear()
        self.video_buffer.clear()
        self.audio_last_update = None
        self.video_last_update = None
        self._previous_video = None

    def health(self) -> list[PipeHealth]:
        running = self.proc is not None and self.proc.poll() is None
        return [
            PipeHealth(
                f"{self.name}_audio",
                running,
                self.audio_buffer.snapshot().size,
                _age(self.audio_last_update),
                self.restarts,
                self.return_code,
                self.error_tail,
                self.audio_buffer.total_samples,
            ),
            PipeHealth(
                f"{self.name}_video",
                running,
                self.video_buffer.snapshot().shape[0],
                _age(self.video_last_update),
                self.restarts,
                self.return_code,
                self.error_tail,
                self.video_buffer.total_rows,
            ),
        ]

    def _stderr_loop(self, proc: subprocess.Popen[bytes] | None) -> None:
        if proc is None or proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            text = _sanitize_error(raw.decode("utf-8", "replace")).strip()
            if text:
                self.error_tail = (self.error_tail + " | " + text)[-500:]


class LiveAnalyzer:
    def __init__(self, source: str, output: str, options: LiveOptions | None = None):
        self.options = options or LiveOptions()
        self.pipes = [
            AVFeaturePipe("source", source, self.options),
            AVFeaturePipe("output", output, self.options),
        ]
        self._last_restart_counts = self._restart_counts()
        self._recent_offsets_ms: list[float] = []
        self._last_stable_result: AlignmentResult | None = None
        self._unknown_estimates = 0

    def start(self) -> None:
        for pipe in self.pipes:
            pipe.start()

    def stop(self) -> None:
        for pipe in self.pipes:
            pipe.stop()

    def estimate(self) -> AlignmentResult:
        self._reset_windows_after_restart()
        source_audio = self.pipes[0].audio_buffer.snapshot()
        output_audio = self.pipes[1].audio_buffer.snapshot()
        source_video = self.pipes[0].video_buffer.snapshot()
        output_video = self.pipes[1].video_buffer.snapshot()
        min_samples = int(self.options.min_overlap_s * self.options.audio_feature_rate)
        min_frames = int(self.options.min_overlap_s * self.options.video_rate)
        blocking_warnings: list[str] = []
        diagnostic_warnings: list[str] = []
        if source_audio.size < min_samples or output_audio.size < min_samples:
            blocking_warnings.append("audio:warming_up")
        if source_video.shape[0] < min_frames or output_video.shape[0] < min_frames:
            blocking_warnings.append("video:warming_up")
        for item in self.health():
            if not item.running and item.samples == 0:
                blocking_warnings.append(f"{item.name}:stopped_no_media")
            if item.restarts:
                diagnostic_warnings.append(f"{item.name}:restarted_{item.restarts}")
        if blocking_warnings:
            return classify_offset(
                audio_latency_s=None,
                video_latency_s=None,
                audio_confidence=0.0,
                video_confidence=0.0,
                warnings=blocking_warnings + diagnostic_warnings,
                ok_ms=self.options.ok_ms,
                warn_ms=self.options.warn_ms,
            )
        result = estimate_alignment(
            source_audio,
            output_audio,
            source_video,
            output_video,
            audio_rate_hz=self.options.audio_feature_rate,
            video_rate_hz=self.options.video_rate,
            min_latency_s=self.options.min_latency_s,
            max_latency_s=self.options.max_latency_s,
            min_overlap_s=self.options.min_overlap_s,
            ok_ms=self.options.ok_ms,
            warn_ms=self.options.warn_ms,
        )
        if not diagnostic_warnings:
            return self._stabilize_result(result)
        result = replace(result, warnings=dedupe_warnings(result.warnings + diagnostic_warnings))
        return self._stabilize_result(result)

    def health(self) -> list[PipeHealth]:
        return [item for pipe in self.pipes for item in pipe.health()]

    def _restart_counts(self) -> tuple[int, ...]:
        return tuple(pipe.restarts for pipe in self.pipes)

    def _reset_windows_after_restart(self) -> None:
        restart_counts = self._restart_counts()
        if restart_counts == self._last_restart_counts:
            return
        for pipe in self.pipes:
            pipe.clear_current_window()
        self._last_restart_counts = restart_counts
        self._recent_offsets_ms.clear()
        self._last_stable_result = None
        self._unknown_estimates = 0

    def _stabilize_result(self, result: AlignmentResult) -> AlignmentResult:
        if not self.options.offset_stability_required:
            return result
        if result.av_offset_ms is None:
            return self._hold_recent_offset(result)

        window = max(1, self.options.offset_stability_window)
        min_estimates = max(1, self.options.offset_stability_min_estimates)
        tolerance_ms = max(0.0, self.options.offset_stability_tolerance_ms)
        self._unknown_estimates = 0
        self._recent_offsets_ms.append(float(result.av_offset_ms))
        self._recent_offsets_ms = self._recent_offsets_ms[-window:]
        if len(self._recent_offsets_ms) < min_estimates:
            return _unstable_offset_result(result)

        offsets = np.asarray(self._recent_offsets_ms, dtype=np.float32)
        median_offset_ms = float(np.median(offsets))
        agreeing = int(np.count_nonzero(np.abs(offsets - median_offset_ms) <= tolerance_ms))
        current_agrees = abs(float(result.av_offset_ms) - median_offset_ms) <= tolerance_ms
        if agreeing >= min_estimates and current_agrees:
            stable = _replace_offset(result, median_offset_ms, ok_ms=self.options.ok_ms, warn_ms=self.options.warn_ms)
            self._last_stable_result = stable
            return stable
        return _unstable_offset_result(result)

    def _hold_recent_offset(self, result: AlignmentResult) -> AlignmentResult:
        if self._last_stable_result is None:
            return result
        self._unknown_estimates += 1
        if self._unknown_estimates > max(0, self.options.offset_hold_unknown_estimates):
            self._recent_offsets_ms.clear()
            self._last_stable_result = None
            return result
        return _held_offset_result(self._last_stable_result, result)


def run_live_analysis(source: str, output: str, *, duration_s: float, options: LiveOptions | None = None) -> tuple[AlignmentResult, list[PipeHealth]]:
    analyzer = LiveAnalyzer(source, output, options)
    analyzer.start()
    try:
        time.sleep(duration_s)
        return analyzer.estimate(), analyzer.health()
    finally:
        analyzer.stop()


def build_av_ffmpeg_cmd(url: str, options: LiveOptions, *, audio_fd: int, video_fd: int) -> list[str]:
    url = normalize_url_arg(url)
    vf = f"fps={options.video_rate},scale={options.video_width}:{options.video_height},format=gray"
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-re",
        "-i",
        url,
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(options.audio_sample_rate),
        "-f",
        "f32le",
        f"pipe:{audio_fd}",
        "-map",
        "0:v:0",
        "-vf",
        vf,
        "-f",
        "rawvideo",
        f"pipe:{video_fd}",
    ]


def _unstable_offset_result(result: AlignmentResult) -> AlignmentResult:
    return replace(
        result,
        verdict="inconclusive",
        direction="unknown",
        av_offset_ms=None,
        warnings=dedupe_warnings(result.warnings + ["unstable_offset", "unreliable_offset"]),
    )


def _held_offset_result(stable: AlignmentResult, current: AlignmentResult) -> AlignmentResult:
    return replace(
        stable,
        verdict="inconclusive",
        warnings=dedupe_warnings(current.warnings + stable.warnings + ["held_offset"]),
    )


def _replace_offset(result: AlignmentResult, offset_ms: float, *, ok_ms: float, warn_ms: float) -> AlignmentResult:
    rounded = round(offset_ms, 3)
    abs_offset = abs(rounded)
    if abs_offset <= ok_ms:
        verdict = "aligned"
        direction = "aligned"
    elif abs_offset <= warn_ms:
        verdict = "warning"
        direction = "audio_ahead" if rounded > 0 else "video_ahead"
    else:
        verdict = "out_of_sync"
        direction = "audio_ahead" if rounded > 0 else "video_ahead"
    return replace(result, verdict=verdict, direction=direction, av_offset_ms=rounded)


def audio_samples_to_features(samples: np.ndarray, *, samples_per_feature: int) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    n = (values.size // samples_per_feature) * samples_per_feature
    if n == 0:
        return np.empty(0, dtype=np.float32)
    framed = values[:n].reshape(-1, samples_per_feature)
    rms = np.sqrt(np.mean(np.square(framed), axis=1))
    peak = np.max(np.abs(framed), axis=1)
    # Log compression keeps speech dynamics but avoids one loud pop dominating.
    return np.log1p((rms * 0.8 + peak * 0.2) * 1000.0).astype(np.float32)


def video_feature_dimensions(*, rows: int = 9, cols: int = 12) -> int:
    return rows * cols * 6


def video_frame_to_feature(
    frame: np.ndarray,
    *,
    previous: np.ndarray | None,
    width: int,
    height: int,
    rows: int = 9,
    cols: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(frame, dtype=np.float32).reshape(height, width)
    prior = current if previous is None else np.asarray(previous, dtype=np.float32).reshape(height, width)
    diff = current - prior
    motion_values: list[float] = []
    appearance_values: list[float] = []
    for row in range(rows):
        y0 = round(row * height / rows)
        y1 = round((row + 1) * height / rows)
        for col in range(cols):
            x0 = round(col * width / cols)
            x1 = round((col + 1) * width / cols)
            diff_block = diff[y0:y1, x0:x1]
            abs_block = np.abs(diff_block)
            motion_values.extend(
                [
                    float(np.mean(abs_block)),
                    float(np.percentile(abs_block, 90)),
                    float(np.mean(diff_block)),
                ]
            )
            current_block = current[y0:y1, x0:x1]
            appearance_values.extend(
                [
                    float(np.mean(current_block)),
                    _mean_abs_diff(current_block, axis=1),
                    _mean_abs_diff(current_block, axis=0),
                ]
            )
    feature = np.concatenate(
        [
            _robust_normalize(np.asarray(motion_values, dtype=np.float32)),
            _robust_normalize(np.asarray(appearance_values, dtype=np.float32)),
        ]
    )
    return feature.astype(np.float32, copy=False), current.reshape(-1).copy()


def normalize_url_arg(url: str) -> str:
    # zsh/bash remove these backslashes when unquoted, but preserve them inside
    # single quotes. Users often paste '?token\=...' after escaping a URL for a
    # different shell context, so repair only URL separators before ffmpeg sees it.
    return url.replace(r"\?", "?").replace(r"\=", "=").replace(r"\&", "&")


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32, copy=False)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < 1e-9:
        return arr - med
    return np.clip((arr - med) / (mad * 6.0), -3.0, 3.0).astype(np.float32)


def _mean_abs_diff(values: np.ndarray, *, axis: int) -> float:
    if values.shape[axis] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(values, axis=axis))))


def _stop_process(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def _age(ts: float | None) -> float | None:
    if ts is None:
        return None
    return round(time.time() - ts, 3)


def _sanitize_error(text: str) -> str:
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"rtmp://\S+", "<rtmp>", text)
    text = re.sub(r"token=[A-Za-z0-9._~+-]+", "token=<redacted>", text)
    return text
