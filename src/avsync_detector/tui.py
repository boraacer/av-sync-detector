from __future__ import annotations

import time

from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .live import LiveAnalyzer, LiveOptions, PipeHealth
from .result import AlignmentResult


def run_tui(source: str, output: str, *, label: str | None, options: LiveOptions, refresh_s: float) -> None:
    analyzer = LiveAnalyzer(source, output, options)
    history: list[float] = []
    analyzer.start()
    started = time.time()
    try:
        with Live(refresh_per_second=max(1, int(1 / max(refresh_s, 0.2))), screen=True, transient=True) as live:
            while True:
                result = analyzer.estimate()
                if result.av_offset_ms is not None:
                    history.append(result.av_offset_ms)
                    history = history[-60:]
                live.update(render_dashboard(result, analyzer.health(), label=label, runtime_s=time.time() - started, history=history))
                time.sleep(refresh_s)
    except KeyboardInterrupt:
        return
    finally:
        analyzer.stop()


def render_dashboard(result: AlignmentResult, health: list[PipeHealth], *, label: str | None, runtime_s: float, history: list[float]) -> Group:
    title = "Live AV Sync Detector" if not label else f"Live AV Sync Detector - {label}"
    return Group(
        Panel(header_text(result, runtime_s), title=title, border_style=verdict_color(result.verdict)),
        Panel(metric_table(result), title="Alignment", border_style=verdict_color(result.verdict)),
        Panel(history_text(history), title="Offset History", border_style="cyan"),
        Panel(health_table(health), title="Decode Health", border_style="blue"),
    )


def header_text(result: AlignmentResult, runtime_s: float) -> Text:
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


def health_table(items: list[PipeHealth]) -> Table:
    table = Table(box=box.SIMPLE, expand=True)
    table.add_column("Pipe")
    table.add_column("State")
    table.add_column("Features", justify="right")
    table.add_column("Restarts", justify="right")
    table.add_column("Last update", justify="right")
    table.add_column("Last error")
    for item in items:
        state = "[green]running[/green]" if item.running else "[red]stopped[/red]"
        age = "never" if item.last_update_age_s is None else f"{item.last_update_age_s:.1f}s ago"
        table.add_row(item.name, state, str(item.samples), str(item.restarts), age, item.error_tail[-100:])
    return table


def history_text(history: list[float]) -> Text:
    if not history:
        return Text("waiting for enough matched content", style="dim")
    chars = "▁▂▃▄▅▆▇█"
    max_abs = max(250.0, max(abs(v) for v in history))
    text = Text()
    for value in history:
        level = min(len(chars) - 1, int(abs(value) / max_abs * (len(chars) - 1)))
        style = "green" if abs(value) <= 120 else "yellow" if abs(value) <= 250 else "red"
        text.append(chars[level], style=style)
    text.append(f"  latest {history[-1]:+.0f}ms", style=verdict_color("aligned" if abs(history[-1]) <= 120 else "warning" if abs(history[-1]) <= 250 else "out_of_sync"))
    return text


def verdict_color(verdict: str) -> str:
    return {
        "aligned": "green",
        "warning": "yellow",
        "out_of_sync": "red",
        "inconclusive": "magenta",
    }.get(verdict, "white")
