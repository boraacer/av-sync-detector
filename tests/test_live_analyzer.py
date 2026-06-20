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
