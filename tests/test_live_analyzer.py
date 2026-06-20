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
