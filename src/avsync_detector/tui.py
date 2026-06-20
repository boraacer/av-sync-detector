from __future__ import annotations

import time
from dataclasses import replace
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .live import LiveAnalyzer, LiveOptions, PipeHealth
from .result import AlignmentResult


def run_tui(source: str, output: str, *, label: str | None, options: LiveOptions | None, refresh_s: float) -> None:
    analyzer_options = replace(options or LiveOptions(), offset_stability_required=True)
    analyzer = LiveAnalyzer(source, output, analyzer_options)
    history: list[tuple[float, float]] = []
    analyzer.start()
    started = time.time()
    try:
        with Live(refresh_per_second=max(1, int(1 / max(refresh_s, 0.2))), screen=True, transient=True) as live:
            while True:
                runtime_s = time.time() - started
                result = analyzer.estimate()
                display_width = _live_display_width(live)
                if result.av_offset_ms is not None:
                    history.append((result.av_offset_ms, runtime_s))
                    history = history[-_history_retention(display_width) :]
                live.update(
                    render_dashboard(
                        result,
                        analyzer.health(),
                        label=label,
                        runtime_s=runtime_s,
                        history=history,
                        source=source,
                        output=output,
                        display_width=display_width,
                    )
                )
                time.sleep(refresh_s)
    except KeyboardInterrupt:
        return
    finally:
        analyzer.stop()


def render_dashboard(
    result: AlignmentResult,
    health: list[PipeHealth],
    *,
    label: str | None,
    runtime_s: float,
    history: list[tuple[float, float]],
    source: str | None = None,
    output: str | None = None,
    display_width: int | None = None,
) -> Group:
    title = "Live AV Sync Detector" if not label else f"Live AV Sync Detector - {label}"
    panels = [
        Panel(header_text(result, runtime_s, source=source, output=output), title=title, border_style=verdict_color(result.verdict)),
    ]
    panels.extend(
        [
        Panel(metric_table(result), title="Alignment", border_style=verdict_color(result.verdict)),
        Panel(
            history_text(
                history,
                current_offset_ms=result.av_offset_ms,
                runtime_s=runtime_s,
                chart_width=_history_chart_width(display_width),
            ),
            title="Offset History",
            border_style="cyan",
        ),
        Panel(health_table(health), title="Decode Health", border_style="blue"),
        ]
    )
    return Group(*panels)


def header_text(result: AlignmentResult, runtime_s: float, *, source: str | None = None, output: str | None = None) -> Text:
    text = Text()
    text.append(f"runtime {runtime_s:0.0f}s  ", style="dim")
    text.append(result.verdict.upper(), style=f"bold {verdict_color(result.verdict)}")
    text.append("  ")
    prefix = "tentative " if result.verdict == "inconclusive" and result.av_offset_ms is not None else ""
    if result.av_offset_ms is None:
        text.append("alignment inconclusive", style="yellow")
    elif result.direction == "aligned":
        text.append(f"{prefix}aligned within {abs(result.av_offset_ms):.0f}ms", style="green")
    elif result.direction == "audio_ahead":
        text.append(f"{prefix}audio ahead by {abs(result.av_offset_ms):.0f}ms", style=verdict_color(result.verdict))
    else:
        text.append(f"{prefix}video ahead by {abs(result.av_offset_ms):.0f}ms", style=verdict_color(result.verdict))
    if source is not None and output is not None:
        text.append("\n")
        text.append("source ", style="dim")
        text.append(sanitize_reference(source), style="cyan")
        text.append("\n")
        text.append("output ", style="dim")
        text.append(sanitize_reference(output), style="cyan")
    return text


def metric_table(result: AlignmentResult) -> Table:
    table = Table(box=box.SIMPLE, expand=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    offset = "unknown" if result.av_offset_ms is None else f"{result.av_offset_ms:+.0f} ms"
    if result.av_offset_ms is None and "unreliable_offset" in result.warnings:
        offset = "unknown (low confidence)"
    table.add_row("A/V offset", offset)
    table.add_row("Audio latency", "unknown" if result.audio_latency_s is None else f"{result.audio_latency_s:.2f} s")
    table.add_row("Video latency", "unknown" if result.video_latency_s is None else f"{result.video_latency_s:.2f} s")
    table.add_row("Mean latency", "unknown" if result.latency_mean_s is None else f"{result.latency_mean_s:.2f} s")
    table.add_row("Audio confidence", f"{result.audio_confidence:.2f}")
    table.add_row("Video confidence", f"{result.video_confidence:.2f}")
    if result.warnings:
        table.add_row("Warnings", ", ".join(result.warnings))
    return table


def sanitize_reference(value: str) -> str:
    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc:
        return value
    netloc = parts.netloc
    if "@" in netloc:
        credentials, host = netloc.rsplit("@", 1)
        username = credentials.split(":", 1)[0]
        netloc = f"{username}:<redacted>@{host}"
    query = ""
    if parts.query:
        query = "&".join(f"{quote(key)}=<redacted>" for key, _ in parse_qsl(parts.query, keep_blank_values=True))
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def health_table(items: list[PipeHealth]) -> Table:
    table = Table(box=box.SIMPLE, expand=True)
    table.add_column("Pipe")
    table.add_column("State")
    table.add_column("Window", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Restarts", justify="right")
    table.add_column("Last update", justify="right")
    table.add_column("Last error")
    for item in items:
        state = "[green]running[/green]" if item.running else "[red]stopped[/red]"
        age = "never" if item.last_update_age_s is None else f"{item.last_update_age_s:.1f}s ago"
        total = item.total_samples if item.total_samples else item.samples
        table.add_row(item.name, state, str(item.samples), str(total), str(item.restarts), age, item.error_tail[-100:])
    return table


def history_text(
    history: list[float] | list[tuple[float, float]],
    *,
    current_offset_ms: float | None = None,
    runtime_s: float | None = None,
    chart_width: int = 80,
    chart_height: int = 4,
) -> Text:
    if not history:
        return Text("waiting for enough matched content", style="dim")
    offsets = [_history_offset(item) for item in history]
    max_abs = max(250.0, max(abs(v) for v in offsets))
    chart_width = max(8, int(chart_width))
    chart_height = max(2, int(chart_height))
    visible_offsets = _sample_history_offsets(offsets, chart_width)
    text = Text()
    for row in range(chart_height):
        threshold = max_abs * (chart_height - row) / chart_height
        for value in visible_offsets:
            if abs(value) >= threshold:
                text.append("█", style=_offset_style(value))
            else:
                text.append(" ")
        text.append("\n")
    latest = visible_offsets[-1]
    latest_style = verdict_color("aligned" if abs(latest) <= 120 else "warning" if abs(latest) <= 250 else "out_of_sync")
    latest_runtime = _history_runtime(history[-1])
    if current_offset_ms is None and runtime_s is not None and latest_runtime is not None:
        age_s = max(0.0, runtime_s - latest_runtime)
        text.append(f"last {latest:+.0f}ms {age_s:.0f}s ago; current unknown", style=latest_style)
    else:
        text.append(f"latest {latest:+.0f}ms", style=latest_style)
    return text


def _sample_history_offsets(offsets: list[float], width: int) -> list[float]:
    if len(offsets) <= width:
        return offsets[-width:]
    start = len(offsets) - width
    return offsets[start:]


def _offset_style(value: float) -> str:
    return "green" if abs(value) <= 120 else "yellow" if abs(value) <= 250 else "red"


def _history_offset(item: float | tuple[float, float]) -> float:
    return item[0] if isinstance(item, tuple) else item


def _history_runtime(item: float | tuple[float, float]) -> float | None:
    return item[1] if isinstance(item, tuple) else None


def _history_chart_width(display_width: int | None) -> int:
    if display_width is None:
        return 80
    return max(20, display_width - 8)


def _history_retention(display_width: int | None) -> int:
    return max(120, _history_chart_width(display_width))


def _live_display_width(live: Live) -> int | None:
    console = getattr(live, "console", None)
    size = getattr(console, "size", None)
    return getattr(size, "width", None)


def verdict_color(verdict: str) -> str:
    return {
        "aligned": "green",
        "warning": "yellow",
        "out_of_sync": "red",
        "inconclusive": "magenta",
    }.get(verdict, "white")
