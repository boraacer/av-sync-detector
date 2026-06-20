from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class DelayEstimate:
    delay_s: float | None
    confidence: float
    score: float
    overlap_s: float
    warnings: list[str]


@dataclass(frozen=True)
class AlignmentResult:
    verdict: str
    direction: str
    av_offset_ms: float | None
    audio_latency_s: float | None
    video_latency_s: float | None
    latency_mean_s: float | None
    audio_confidence: float
    video_confidence: float
    overall_confidence: float
    warnings: list[str]


def classify_offset(
    *,
    audio_latency_s: float | None,
    video_latency_s: float | None,
    audio_confidence: float,
    video_confidence: float,
    warnings: list[str] | None = None,
    ok_ms: float = 120.0,
    warn_ms: float = 250.0,
    min_confidence: float = 0.2,
) -> AlignmentResult:
    merged_warnings = list(warnings or [])
    overall = max(0.0, min(1.0, min(audio_confidence, video_confidence)))
    if (
        audio_latency_s is None
        or video_latency_s is None
        or not isfinite(audio_latency_s)
        or not isfinite(video_latency_s)
    ):
        return AlignmentResult(
            verdict="inconclusive",
            direction="unknown",
            av_offset_ms=None,
            audio_latency_s=audio_latency_s,
            video_latency_s=video_latency_s,
            latency_mean_s=None,
            audio_confidence=audio_confidence,
            video_confidence=video_confidence,
            overall_confidence=overall,
            warnings=dedupe_warnings(merged_warnings + ["missing_delay_estimate"]),
        )
    offset_ms = (video_latency_s - audio_latency_s) * 1000.0
    abs_offset = abs(offset_ms)
    if abs_offset <= ok_ms:
        direction = "aligned"
    else:
        direction = "audio_ahead" if offset_ms > 0 else "video_ahead"

    if overall < min_confidence:
        return AlignmentResult(
            verdict="inconclusive",
            direction="unknown",
            av_offset_ms=None,
            audio_latency_s=round(audio_latency_s, 6),
            video_latency_s=round(video_latency_s, 6),
            latency_mean_s=round((audio_latency_s + video_latency_s) / 2, 6),
            audio_confidence=round(audio_confidence, 6),
            video_confidence=round(video_confidence, 6),
            overall_confidence=round(overall, 6),
            warnings=dedupe_warnings(merged_warnings + ["low_confidence", "unreliable_offset"]),
        )

    if abs_offset <= ok_ms:
        verdict = "aligned"
    elif abs_offset <= warn_ms:
        verdict = "warning"
    else:
        verdict = "out_of_sync"

    return AlignmentResult(
        verdict=verdict,
        direction=direction,
        av_offset_ms=round(offset_ms, 3),
        audio_latency_s=round(audio_latency_s, 6),
        video_latency_s=round(video_latency_s, 6),
        latency_mean_s=round((audio_latency_s + video_latency_s) / 2, 6),
        audio_confidence=round(audio_confidence, 6),
        video_confidence=round(video_confidence, 6),
        overall_confidence=round(overall, 6),
        warnings=dedupe_warnings(merged_warnings),
    )


def dedupe_warnings(warnings: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for warning in warnings:
        if warning and warning not in seen:
            seen.add(warning)
            out.append(warning)
    return out
