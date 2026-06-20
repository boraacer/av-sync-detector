import json

from avsync_detector.cli import build_parser, options_from_args, result_to_json_dict
from avsync_detector.result import AlignmentResult


def test_json_subcommand_requires_only_source_and_output():
    parser = build_parser()

    args = parser.parse_args(["json", "--source", "http://source.test/live", "--output", "http://output.test/live"])

    assert args.command == "json"
    assert args.source == "http://source.test/live"
    assert args.output == "http://output.test/live"
    assert args.label is None
    assert args.duration == 30
    assert args.fail_on_bad is False


def test_tui_is_default_command():
    parser = build_parser()

    args = parser.parse_args(["--source", "http://source.test/live", "--output", "http://output.test/live"])

    assert args.command == "tui"
    assert args.source == "http://source.test/live"
    assert args.output == "http://output.test/live"
    assert args.refresh == 3


def test_result_json_reports_alignment_not_latency_as_verdict():
    result = AlignmentResult(
        verdict="aligned",
        direction="aligned",
        av_offset_ms=40.0,
        audio_latency_s=31.2,
        video_latency_s=31.24,
        latency_mean_s=31.22,
        audio_confidence=0.91,
        video_confidence=0.82,
        overall_confidence=0.82,
        warnings=[],
    )

    payload = result_to_json_dict(result, duration_s=30)

    assert payload["verdict"] == "aligned"
    assert payload["direction"] == "aligned"
    assert payload["av_offset_ms"] == 40.0
    assert payload["latency"]["mean_s"] == 31.22
    assert "stream" not in payload
    json.dumps(payload)


def test_video_rate_is_high_enough_by_default_and_configurable():
    parser = build_parser()

    default_args = parser.parse_args(["json", "--source", "http://source.test/live", "--output", "http://output.test/live"])
    custom_args = parser.parse_args(
        ["json", "--source", "http://source.test/live", "--output", "http://output.test/live", "--video-rate", "25"]
    )

    assert options_from_args(default_args).video_rate == 20
    assert options_from_args(custom_args).video_rate == 25


def test_default_latency_search_covers_hls_distribution_delay():
    parser = build_parser()

    args = parser.parse_args(["tui", "--source", "http://source.test/live", "--output", "http://output.test/live"])

    assert options_from_args(args).max_latency_s == 60
