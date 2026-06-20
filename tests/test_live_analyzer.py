import numpy as np

from avsync_detector.result import AlignmentResult
from avsync_detector.live import LiveAnalyzer, LiveOptions


def test_live_analyzer_uses_one_capture_session_per_url():
    analyzer = LiveAnalyzer("source", "output", LiveOptions())

    assert len(analyzer.pipes) == 2
    assert [pipe.name for pipe in analyzer.pipes] == ["source", "output"]
    assert [item.name for item in analyzer.health()] == [
        "source_audio",
        "source_video",
        "output_audio",
        "output_video",
    ]


def test_pipe_health_reports_total_features_beyond_rolling_window():
    options = LiveOptions(rolling_window_s=1, audio_feature_rate=2, video_rate=1)
    analyzer = LiveAnalyzer("source", "output", options)
    pipe = analyzer.pipes[0]

    pipe.audio_buffer.append(np.asarray([1, 2, 3], dtype=np.float32))
    pipe.video_buffer.append(np.zeros(pipe.video_buffer.dims, dtype=np.float32))
    pipe.video_buffer.append(np.ones(pipe.video_buffer.dims, dtype=np.float32))

    audio_health, video_health = pipe.health()

    assert audio_health.samples == 2
    assert audio_health.total_samples == 3
    assert video_health.samples == 1
    assert video_health.total_samples == 2


def test_any_capture_restart_resets_all_current_windows():
    options = LiveOptions(min_overlap_s=1, audio_feature_rate=10, video_rate=2)
    analyzer = LiveAnalyzer("source", "output", options)
    for pipe in analyzer.pipes:
        pipe.audio_buffer.append(np.asarray([1, 2, 3], dtype=np.float32))
        pipe.video_buffer.append(np.ones(pipe.video_buffer.dims, dtype=np.float32))

    analyzer.pipes[0].restarts = 1

    result = analyzer.estimate()

    for pipe in analyzer.pipes:
        assert pipe.audio_buffer.snapshot().size == 0
        assert pipe.video_buffer.snapshot().shape[0] == 0
    assert result.av_offset_ms is None
    assert "audio:warming_up" in result.warnings
    assert "video:warming_up" in result.warnings
    assert "source_audio:restarted_1" in result.warnings


def test_restart_warning_does_not_permanently_block_measurement(monkeypatch):
    options = LiveOptions(min_overlap_s=1, audio_feature_rate=10, video_rate=2)
    analyzer = LiveAnalyzer("source", "output", options)
    for pipe in analyzer.pipes:
        pipe.audio_buffer.append(np.asarray([1, 2, 4, 8, 4, 2, 1, 2, 4, 8], dtype=np.float32))
        for value in range(2):
            pipe.video_buffer.append(np.full(pipe.video_buffer.dims, value % 2, dtype=np.float32))
        pipe.restarts = 1
    analyzer._last_restart_counts = analyzer._restart_counts()

    def fake_estimate_alignment(*args, **kwargs):
        return AlignmentResult(
            verdict="aligned",
            direction="aligned",
            av_offset_ms=10.0,
            audio_latency_s=2.0,
            video_latency_s=2.01,
            latency_mean_s=2.005,
            audio_confidence=0.9,
            video_confidence=0.8,
            overall_confidence=0.8,
            warnings=[],
        )

    monkeypatch.setattr("avsync_detector.live.estimate_alignment", fake_estimate_alignment)

    result = analyzer.estimate()

    assert result.av_offset_ms == 10.0
    assert "source_audio:restarted_1" in result.warnings
    assert "output_video:restarted_1" in result.warnings


def test_live_analyzer_hides_volatile_offsets_until_recent_estimates_agree(monkeypatch):
    options = LiveOptions(
        min_overlap_s=1,
        audio_feature_rate=10,
        video_rate=2,
        offset_stability_required=True,
        offset_stability_min_estimates=3,
        offset_stability_tolerance_ms=80.0,
    )
    analyzer = LiveAnalyzer("source", "output", options)
    _fill_analyzer_windows(analyzer)
    offsets = iter([20.0, -260.0, 210.0])

    def fake_estimate_alignment(*args, **kwargs):
        offset_ms = next(offsets)
        return _alignment(offset_ms)

    monkeypatch.setattr("avsync_detector.live.estimate_alignment", fake_estimate_alignment)

    first = analyzer.estimate()
    second = analyzer.estimate()
    third = analyzer.estimate()

    assert first.av_offset_ms is None
    assert second.av_offset_ms is None
    assert third.av_offset_ms is None
    assert third.verdict == "inconclusive"
    assert "unstable_offset" in third.warnings
    assert "unreliable_offset" in third.warnings


def test_live_analyzer_publishes_median_offset_after_recent_estimates_agree(monkeypatch):
    options = LiveOptions(
        min_overlap_s=1,
        audio_feature_rate=10,
        video_rate=2,
        offset_stability_required=True,
        offset_stability_min_estimates=3,
        offset_stability_tolerance_ms=80.0,
    )
    analyzer = LiveAnalyzer("source", "output", options)
    _fill_analyzer_windows(analyzer)
    offsets = iter([15.0, 35.0, 20.0])

    def fake_estimate_alignment(*args, **kwargs):
        offset_ms = next(offsets)
        return _alignment(offset_ms)

    monkeypatch.setattr("avsync_detector.live.estimate_alignment", fake_estimate_alignment)

    analyzer.estimate()
    analyzer.estimate()
    result = analyzer.estimate()

    assert result.verdict == "aligned"
    assert result.direction == "aligned"
    assert result.av_offset_ms == 20.0
    assert "unstable_offset" not in result.warnings


def test_live_analyzer_holds_last_stable_offset_through_brief_unknown(monkeypatch):
    options = LiveOptions(
        min_overlap_s=1,
        audio_feature_rate=10,
        video_rate=2,
        offset_stability_required=True,
        offset_stability_min_estimates=3,
        offset_stability_tolerance_ms=80.0,
        offset_hold_unknown_estimates=2,
    )
    analyzer = LiveAnalyzer("source", "output", options)
    _fill_analyzer_windows(analyzer)
    results = iter(
        [
            _alignment(20.0),
            _alignment(25.0),
            _alignment(15.0),
            _unknown_alignment(["video:low_signal"]),
            _alignment(30.0),
        ]
    )

    def fake_estimate_alignment(*args, **kwargs):
        return next(results)

    monkeypatch.setattr("avsync_detector.live.estimate_alignment", fake_estimate_alignment)

    analyzer.estimate()
    analyzer.estimate()
    stable = analyzer.estimate()
    held = analyzer.estimate()
    resumed = analyzer.estimate()

    assert stable.av_offset_ms == 20.0
    assert held.verdict == "inconclusive"
    assert held.direction == "aligned"
    assert held.av_offset_ms == 20.0
    assert "held_offset" in held.warnings
    assert "video:low_signal" in held.warnings
    assert resumed.av_offset_ms == 22.5
    assert "unstable_offset" not in resumed.warnings


def test_live_analyzer_stops_holding_offset_after_repeated_unknowns(monkeypatch):
    options = LiveOptions(
        min_overlap_s=1,
        audio_feature_rate=10,
        video_rate=2,
        offset_stability_required=True,
        offset_stability_min_estimates=3,
        offset_stability_tolerance_ms=80.0,
        offset_hold_unknown_estimates=1,
    )
    analyzer = LiveAnalyzer("source", "output", options)
    _fill_analyzer_windows(analyzer)
    results = iter(
        [
            _alignment(20.0),
            _alignment(25.0),
            _alignment(15.0),
            _unknown_alignment(["video:low_signal"]),
            _unknown_alignment(["video:low_signal"]),
        ]
    )

    def fake_estimate_alignment(*args, **kwargs):
        return next(results)

    monkeypatch.setattr("avsync_detector.live.estimate_alignment", fake_estimate_alignment)

    analyzer.estimate()
    analyzer.estimate()
    analyzer.estimate()
    held = analyzer.estimate()
    expired = analyzer.estimate()

    assert held.av_offset_ms == 20.0
    assert "held_offset" in held.warnings
    assert expired.av_offset_ms is None
    assert "held_offset" not in expired.warnings


def _fill_analyzer_windows(analyzer: LiveAnalyzer) -> None:
    for pipe in analyzer.pipes:
        pipe.audio_buffer.append(np.asarray([1, 2, 4, 8, 4, 2, 1, 2, 4, 8], dtype=np.float32))
        for value in range(2):
            pipe.video_buffer.append(np.full(pipe.video_buffer.dims, value % 2, dtype=np.float32))


def _alignment(offset_ms: float) -> AlignmentResult:
    audio_latency_s = 2.0
    video_latency_s = audio_latency_s + offset_ms / 1000.0
    return AlignmentResult(
        verdict="aligned" if abs(offset_ms) <= 120 else "out_of_sync",
        direction="aligned" if abs(offset_ms) <= 120 else "audio_ahead" if offset_ms > 0 else "video_ahead",
        av_offset_ms=offset_ms,
        audio_latency_s=audio_latency_s,
        video_latency_s=video_latency_s,
        latency_mean_s=(audio_latency_s + video_latency_s) / 2,
        audio_confidence=0.9,
        video_confidence=0.8,
        overall_confidence=0.8,
        warnings=[],
    )


def _unknown_alignment(warnings: list[str]) -> AlignmentResult:
    return AlignmentResult(
        verdict="inconclusive",
        direction="unknown",
        av_offset_ms=None,
        audio_latency_s=None,
        video_latency_s=None,
        latency_mean_s=None,
        audio_confidence=0.0,
        video_confidence=0.0,
        overall_confidence=0.0,
        warnings=warnings,
    )
