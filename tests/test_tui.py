from avsync_detector.result import AlignmentResult
from avsync_detector.live import PipeHealth
from avsync_detector.tui import header_text, health_table, history_text, metric_table, run_tui
from rich.console import Console


def test_tui_uses_full_screen_live(monkeypatch):
    calls = {}

    class FakeAnalyzer:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def estimate(self):
            return AlignmentResult(
                verdict="inconclusive",
                direction="unknown",
                av_offset_ms=None,
                audio_latency_s=None,
                video_latency_s=None,
                latency_mean_s=None,
                audio_confidence=0,
                video_confidence=0,
                overall_confidence=0,
                warnings=["warming_up"],
            )

        def health(self):
            return []

    class FakeLive:
        def __init__(self, *args, **kwargs):
            calls["kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return exc_type is KeyboardInterrupt

        def update(self, _renderable):
            raise KeyboardInterrupt

    monkeypatch.setattr("avsync_detector.tui.LiveAnalyzer", FakeAnalyzer)
    monkeypatch.setattr("avsync_detector.tui.Live", FakeLive)

    run_tui("source", "output", label=None, options=None, refresh_s=1)

    assert calls["kwargs"]["screen"] is True
    assert calls["kwargs"]["transient"] is True


def test_header_keeps_low_confidence_offset_unknown():
    result = AlignmentResult(
        verdict="inconclusive",
        direction="unknown",
        av_offset_ms=None,
        audio_latency_s=18.5,
        video_latency_s=24.0,
        latency_mean_s=21.25,
        audio_confidence=0.03,
        video_confidence=0.02,
        overall_confidence=0.02,
        warnings=["low_confidence", "unreliable_offset"],
    )

    text = header_text(result, runtime_s=64).plain

    assert "INCONCLUSIVE" in text
    assert "alignment inconclusive" in text


def test_metric_table_explains_untrusted_offset_candidate():
    result = AlignmentResult(
        verdict="inconclusive",
        direction="unknown",
        av_offset_ms=None,
        audio_latency_s=15.5,
        video_latency_s=24.8,
        latency_mean_s=20.15,
        audio_confidence=0.05,
        video_confidence=0.0,
        overall_confidence=0.0,
        warnings=["low_confidence", "unreliable_offset"],
    )
    console = Console(record=True, width=120, color_system=None)

    console.print(metric_table(result))

    assert "unknown (low confidence)" in console.export_text()


def test_health_table_shows_rolling_window_and_total_features():
    console = Console(record=True, width=120, color_system=None)

    console.print(
        health_table(
            [
                PipeHealth(
                    name="source_audio",
                    running=True,
                    samples=12000,
                    last_update_age_s=0.0,
                    restarts=3,
                    error_tail="decode error",
                    total_samples=48500,
                )
            ]
        )
    )

    text = console.export_text()
    assert "Window" in text
    assert "Total" in text
    assert "12000" in text
    assert "48500" in text


def test_history_marks_previous_offset_stale_when_current_result_is_unknown():
    text = history_text(
        [(-150.0, 43.0)],
        current_offset_ms=None,
        runtime_s=1134.0,
    ).plain

    assert "last -150ms" in text
    assert "1091s ago" in text
    assert "current unknown" in text
    assert "latest" not in text
