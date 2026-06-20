from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .live import LiveOptions, run_live_analysis
from .result import AlignmentResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avsync-detector", description="Live A/V alignment detector using content similarity.")
    sub = parser.add_subparsers(dest="command", required=True)

    json_parser = sub.add_parser("json", help="Run once and print one JSON result.")
    add_common_args(json_parser)
    json_parser.add_argument("--duration", type=float, default=30, help="Seconds to observe before printing JSON.")
    json_parser.add_argument("--fail-on-bad", action="store_true", help="Exit nonzero when verdict is out_of_sync or inconclusive.")

    tui_parser = sub.add_parser("tui", help="Run an interactive colored terminal dashboard.")
    add_common_args(tui_parser)
    tui_parser.add_argument("--refresh", type=float, default=3, help="TUI refresh interval in seconds.")
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, help="Live source URL.")
    parser.add_argument("--output", required=True, help="Live output URL.")
    parser.add_argument("--label", default=None, help="Optional display label only.")
    parser.add_argument("--min-latency", type=float, default=0.0, help="Minimum source-to-output latency to search, in seconds.")
    parser.add_argument("--max-latency", type=float, default=60.0, help="Maximum source-to-output latency to search, in seconds.")
    parser.add_argument("--min-overlap", type=float, default=6.0, help="Minimum matched content overlap, in seconds.")
    parser.add_argument("--window", type=float, default=120.0, help="Rolling RAM window in seconds.")
    parser.add_argument("--video-rate", type=int, default=20, help="Video fingerprint sample rate in frames per second.")
    parser.add_argument("--ok-ms", type=float, default=120.0, help="Aligned threshold in milliseconds.")
    parser.add_argument("--warn-ms", type=float, default=250.0, help="Warning threshold in milliseconds.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    options = options_from_args(args)
    if args.command == "json":
        result, health = run_live_analysis(args.source, args.output, duration_s=args.duration, options=options)
        payload = result_to_json_dict(result, duration_s=args.duration, label=args.label)
        payload["health"] = [asdict(item) for item in health]
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.fail_on_bad and result.verdict not in {"aligned", "warning"}:
            return 2
        return 0
    if args.command == "tui":
        from .tui import run_tui

        run_tui(args.source, args.output, label=args.label, options=options, refresh_s=args.refresh)
        return 0
    parser.error(f"unknown command {args.command}")
    return 2


def options_from_args(args: argparse.Namespace) -> LiveOptions:
    return LiveOptions(
        min_latency_s=args.min_latency,
        max_latency_s=args.max_latency,
        min_overlap_s=args.min_overlap,
        rolling_window_s=args.window,
        video_rate=args.video_rate,
        ok_ms=args.ok_ms,
        warn_ms=args.warn_ms,
    )


def result_to_json_dict(result: AlignmentResult, *, duration_s: float, label: str | None = None) -> dict:
    payload = {
        "duration_s": duration_s,
        "verdict": result.verdict,
        "direction": result.direction,
        "av_offset_ms": result.av_offset_ms,
        "latency": {
            "audio_s": result.audio_latency_s,
            "video_s": result.video_latency_s,
            "mean_s": result.latency_mean_s,
        },
        "confidence": {
            "audio": result.audio_confidence,
            "video": result.video_confidence,
            "overall": result.overall_confidence,
        },
        "warnings": result.warnings,
    }
    if label:
        payload["label"] = label
    return payload


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
