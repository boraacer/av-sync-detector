import numpy as np

from avsync_detector.align import estimate_alignment, estimate_delay_1d, estimate_delay_vectors
from avsync_detector.result import classify_offset


def pulse_train(length: int, period: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.02, length)
    for idx in range(period, length, period):
        width = 3 + (idx // period) % 4
        x[idx : idx + width] += np.linspace(1.0, 0.25, width)
    return x.astype(np.float32)


def delayed_1d(x: np.ndarray, delay: int, noise: float = 0.0) -> np.ndarray:
    y = np.pad(x, (delay, 0), mode="constant")[: len(x)]
    if noise:
        y = y + np.random.default_rng(11).normal(0, noise, len(y))
    return y.astype(np.float32)


def vector_features(length: int, dims: int = 16, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (length, dims)).astype(np.float32)
    # Add smooth motion so adjacent frames are related but still matchable.
    for i in range(1, length):
        base[i] = base[i] * 0.35 + base[i - 1] * 0.65
    return base


def delayed_vectors(x: np.ndarray, delay: int, noise: float = 0.0) -> np.ndarray:
    pad = np.zeros((delay, x.shape[1]), dtype=np.float32)
    y = np.vstack([pad, x])[: len(x)]
    if noise:
        y = y + np.random.default_rng(23).normal(0, noise, y.shape)
    return y.astype(np.float32)


def test_equal_large_latency_is_aligned_even_when_delay_is_large():
    audio_rate = 100
    video_rate = 10
    audio_delay_s = 16.4
    video_delay_s = 16.4

    source_audio = pulse_train(5_000, 47)
    output_audio = delayed_1d(source_audio, int(audio_delay_s * audio_rate), noise=0.01)
    source_video = vector_features(500)
    output_video = delayed_vectors(source_video, int(video_delay_s * video_rate), noise=0.01)

    result = estimate_alignment(
        source_audio,
        output_audio,
        source_video,
        output_video,
        audio_rate_hz=audio_rate,
        video_rate_hz=video_rate,
        min_latency_s=10,
        max_latency_s=25,
    )

    assert result.verdict == "aligned"
    assert abs(result.av_offset_ms) <= 120
    assert abs(result.audio_latency_s - audio_delay_s) <= 0.02
    assert abs(result.video_latency_s - video_delay_s) <= 0.11


def test_video_late_is_reported_as_audio_ahead():
    audio_rate = 100
    video_rate = 10
    source_audio = pulse_train(6_000, 53)
    source_video = vector_features(600)

    result = estimate_alignment(
        source_audio,
        delayed_1d(source_audio, int(15.0 * audio_rate), noise=0.01),
        source_video,
        delayed_vectors(source_video, int(15.8 * video_rate), noise=0.01),
        audio_rate_hz=audio_rate,
        video_rate_hz=video_rate,
        min_latency_s=10,
        max_latency_s=22,
    )

    assert result.verdict == "out_of_sync"
    assert result.direction == "audio_ahead"
    assert 700 <= result.av_offset_ms <= 900


def test_static_video_has_low_confidence_and_alignment_is_inconclusive():
    static_source = np.zeros((400, 16), dtype=np.float32)
    static_output = np.zeros((400, 16), dtype=np.float32)

    estimate = estimate_delay_vectors(
        static_source,
        static_output,
        rate_hz=10,
        min_latency_s=0,
        max_latency_s=20,
    )

    assert estimate.confidence < 0.2
    assert "low_signal" in estimate.warnings


def test_classify_offset_ignores_large_equal_latency():
    result = classify_offset(audio_latency_s=25.0, video_latency_s=25.05, audio_confidence=0.9, video_confidence=0.8)

    assert result.verdict == "aligned"
    assert result.direction == "aligned"
    assert result.latency_mean_s == 25.025
    assert abs(result.av_offset_ms - 50) < 0.001


def test_classify_offset_rejects_low_confidence_offset_candidate():
    result = classify_offset(audio_latency_s=18.5, video_latency_s=24.0, audio_confidence=0.03, video_confidence=0.02)

    assert result.verdict == "inconclusive"
    assert result.direction == "unknown"
    assert result.av_offset_ms is None
    assert result.latency_mean_s == 21.25
    assert "low_confidence" in result.warnings
    assert "unreliable_offset" in result.warnings


def test_classify_offset_rejects_zero_confidence_offset_candidate():
    result = classify_offset(audio_latency_s=15.5, video_latency_s=24.8, audio_confidence=0.05, video_confidence=0.0)

    assert result.verdict == "inconclusive"
    assert result.direction == "unknown"
    assert result.av_offset_ms is None
    assert result.audio_latency_s == 15.5
    assert result.video_latency_s == 24.8
    assert result.latency_mean_s == 20.15
    assert "unreliable_offset" in result.warnings


def test_classify_offset_keeps_missing_estimate_unknown():
    result = classify_offset(audio_latency_s=None, video_latency_s=24.0, audio_confidence=0.03, video_confidence=0.8)

    assert result.verdict == "inconclusive"
    assert result.direction == "unknown"
    assert result.av_offset_ms is None
    assert result.latency_mean_s is None
    assert "missing_delay_estimate" in result.warnings


def test_audio_delay_estimate_reports_low_signal_for_silence():
    estimate = estimate_delay_1d(np.zeros(1_000), np.zeros(1_000), rate_hz=100, min_latency_s=0, max_latency_s=5)

    assert estimate.confidence == 0
    assert "low_signal" in estimate.warnings


def test_delay_estimate_rejects_match_on_max_latency_boundary():
    source = pulse_train(6_000, 47)
    output = delayed_1d(source, 2_500, noise=0.01)

    estimate = estimate_delay_1d(
        source,
        output,
        rate_hz=100,
        min_latency_s=0,
        max_latency_s=25,
        min_overlap_s=6,
    )

    assert estimate.delay_s == 25
    assert estimate.confidence == 0
    assert "max_latency_boundary" in estimate.warnings


def test_delay_estimate_rejects_match_on_min_latency_boundary():
    source = pulse_train(6_000, 47)

    estimate = estimate_delay_1d(
        source,
        source.copy(),
        rate_hz=100,
        min_latency_s=0,
        max_latency_s=25,
        min_overlap_s=6,
    )

    assert estimate.delay_s == 0
    assert estimate.confidence == 0
    assert "min_latency_boundary" in estimate.warnings
