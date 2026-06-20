from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .result import AlignmentResult, DelayEstimate, classify_offset, dedupe_warnings


def estimate_alignment(
    source_audio: np.ndarray,
    output_audio: np.ndarray,
    source_video: np.ndarray,
    output_video: np.ndarray,
    *,
    audio_rate_hz: float,
    video_rate_hz: float,
    min_latency_s: float = 0.0,
    max_latency_s: float = 25.0,
    min_overlap_s: float = 6.0,
    ok_ms: float = 120.0,
    warn_ms: float = 250.0,
) -> AlignmentResult:
    audio = estimate_delay_1d(
        source_audio,
        output_audio,
        rate_hz=audio_rate_hz,
        min_latency_s=min_latency_s,
        max_latency_s=max_latency_s,
        min_overlap_s=min_overlap_s,
    )
    video = estimate_delay_vectors(
        source_video,
        output_video,
        rate_hz=video_rate_hz,
        min_latency_s=min_latency_s,
        max_latency_s=max_latency_s,
        min_overlap_s=min_overlap_s,
    )
    warnings = dedupe_warnings(["audio:" + w for w in audio.warnings] + ["video:" + w for w in video.warnings])
    return classify_offset(
        audio_latency_s=audio.delay_s,
        video_latency_s=video.delay_s,
        audio_confidence=audio.confidence,
        video_confidence=video.confidence,
        warnings=warnings,
        ok_ms=ok_ms,
        warn_ms=warn_ms,
    )


def estimate_delay_1d(
    source: np.ndarray,
    output: np.ndarray,
    *,
    rate_hz: float,
    min_latency_s: float = 0.0,
    max_latency_s: float = 25.0,
    min_overlap_s: float = 6.0,
) -> DelayEstimate:
    src = np.asarray(source, dtype=np.float32).reshape(-1)
    out = np.asarray(output, dtype=np.float32).reshape(-1)
    if src.size == 0 or out.size == 0:
        return DelayEstimate(None, 0.0, 0.0, 0.0, ["empty_signal"])
    if _low_signal_1d(src) or _low_signal_1d(out):
        return DelayEstimate(None, 0.0, 0.0, 0.0, ["low_signal"])

    return _search_lag(
        src,
        out,
        rate_hz=rate_hz,
        min_latency_s=min_latency_s,
        max_latency_s=max_latency_s,
        min_overlap_s=min_overlap_s,
        scorer=_score_1d,
    )


def estimate_delay_vectors(
    source: np.ndarray,
    output: np.ndarray,
    *,
    rate_hz: float,
    min_latency_s: float = 0.0,
    max_latency_s: float = 25.0,
    min_overlap_s: float = 6.0,
) -> DelayEstimate:
    src = np.asarray(source, dtype=np.float32)
    out = np.asarray(output, dtype=np.float32)
    if src.ndim != 2 or out.ndim != 2 or src.shape[0] == 0 or out.shape[0] == 0:
        return DelayEstimate(None, 0.0, 0.0, 0.0, ["empty_signal"])
    if src.shape[1] != out.shape[1]:
        return DelayEstimate(None, 0.0, 0.0, 0.0, ["feature_dimension_mismatch"])
    if _low_signal_vectors(src) or _low_signal_vectors(out):
        return DelayEstimate(None, 0.0, 0.0, 0.0, ["low_signal"])

    return _search_lag_vectors(
        src,
        out,
        rate_hz=rate_hz,
        min_latency_s=min_latency_s,
        max_latency_s=max_latency_s,
        min_overlap_s=min_overlap_s,
    )


def _search_lag(
    source: np.ndarray,
    output: np.ndarray,
    *,
    rate_hz: float,
    min_latency_s: float,
    max_latency_s: float,
    min_overlap_s: float,
    scorer,
) -> DelayEstimate:
    min_lag = max(0, int(round(min_latency_s * rate_hz)))
    max_lag = max(min_lag, int(round(max_latency_s * rate_hz)))
    min_overlap = max(1, int(round(min_overlap_s * rate_hz)))
    max_lag = min(max_lag, max(0, output.shape[0] - min_overlap), max(0, source.shape[0] - min_overlap))
    if max_lag < min_lag:
        return DelayEstimate(None, 0.0, 0.0, 0.0, ["insufficient_overlap"])

    best_lag: int | None = None
    best_score = -math.inf
    second_score = -math.inf
    best_overlap = 0
    exclusion = max(1, int(round(0.5 * rate_hz)))
    scored: list[tuple[int, float, int]] = []

    for lag in range(min_lag, max_lag + 1):
        overlap = min(source.shape[0], output.shape[0] - lag)
        if overlap < min_overlap:
            continue
        score = scorer(source[:overlap], output[lag : lag + overlap])
        if not math.isfinite(score):
            continue
        scored.append((lag, score, overlap))
        if score > best_score:
            best_lag = lag
            best_score = score
            best_overlap = overlap

    if best_lag is None:
        return DelayEstimate(None, 0.0, 0.0, 0.0, ["no_match"])

    for lag, score, _ in scored:
        if abs(lag - best_lag) > exclusion and score > second_score:
            second_score = score
    dominance = 0.0 if not math.isfinite(second_score) else max(0.0, best_score - second_score)
    confidence = _confidence_from_score(best_score, dominance)
    warnings: list[str] = []
    if best_lag == min_lag:
        confidence = 0.0
        warnings.append("min_latency_boundary")
    elif best_lag == max_lag:
        confidence = 0.0
        warnings.append("max_latency_boundary")
    if confidence < 0.2:
        warnings.append("low_confidence")
    return DelayEstimate(
        delay_s=round(best_lag / rate_hz, 6),
        confidence=round(confidence, 6),
        score=round(float(best_score), 6),
        overlap_s=round(best_overlap / rate_hz, 6),
        warnings=warnings,
    )


@dataclass(frozen=True)
class _VectorSearchStats:
    source: np.ndarray
    output: np.ndarray
    source_sum: np.ndarray
    source_sq_sum: np.ndarray
    output_sum: np.ndarray
    output_sq_sum: np.ndarray

    def __init__(self, source: np.ndarray, output: np.ndarray):
        src = np.asarray(source, dtype=np.float32)
        out = np.asarray(output, dtype=np.float32)
        object.__setattr__(self, "source", src)
        object.__setattr__(self, "output", out)
        object.__setattr__(self, "source_sum", _prefix_sum(src))
        object.__setattr__(self, "source_sq_sum", _prefix_sum(np.square(src)))
        object.__setattr__(self, "output_sum", _prefix_sum(out))
        object.__setattr__(self, "output_sq_sum", _prefix_sum(np.square(out)))


def _search_lag_vectors(
    source: np.ndarray,
    output: np.ndarray,
    *,
    rate_hz: float,
    min_latency_s: float,
    max_latency_s: float,
    min_overlap_s: float,
) -> DelayEstimate:
    min_lag = max(0, int(round(min_latency_s * rate_hz)))
    max_lag = max(min_lag, int(round(max_latency_s * rate_hz)))
    min_overlap = max(1, int(round(min_overlap_s * rate_hz)))
    max_lag = min(max_lag, max(0, output.shape[0] - min_overlap), max(0, source.shape[0] - min_overlap))
    if max_lag < min_lag:
        return DelayEstimate(None, 0.0, 0.0, 0.0, ["insufficient_overlap"])

    stats = _VectorSearchStats(source, output)
    best_lag: int | None = None
    best_score = -math.inf
    second_score = -math.inf
    best_overlap = 0
    exclusion = max(1, int(round(0.5 * rate_hz)))
    scored: list[tuple[int, float, int]] = []

    for lag in range(min_lag, max_lag + 1):
        overlap = min(source.shape[0], output.shape[0] - lag)
        if overlap < min_overlap:
            continue
        score = _score_vectors_from_stats(stats, lag=lag, overlap=overlap)
        if not math.isfinite(score):
            continue
        scored.append((lag, score, overlap))
        if score > best_score:
            best_lag = lag
            best_score = score
            best_overlap = overlap

    if best_lag is None:
        return DelayEstimate(None, 0.0, 0.0, 0.0, ["no_match"])

    for lag, score, _ in scored:
        if abs(lag - best_lag) > exclusion and score > second_score:
            second_score = score
    dominance = 0.0 if not math.isfinite(second_score) else max(0.0, best_score - second_score)
    confidence = _confidence_from_score(best_score, dominance)
    warnings: list[str] = []
    if best_lag == min_lag:
        confidence = 0.0
        warnings.append("min_latency_boundary")
    elif best_lag == max_lag:
        confidence = 0.0
        warnings.append("max_latency_boundary")
    if confidence < 0.2:
        warnings.append("low_confidence")
    return DelayEstimate(
        delay_s=round(best_lag / rate_hz, 6),
        confidence=round(confidence, 6),
        score=round(float(best_score), 6),
        overlap_s=round(best_overlap / rate_hz, 6),
        warnings=warnings,
    )


def _score_1d(a: np.ndarray, b: np.ndarray) -> float:
    az = _zscore(a)
    bz = _zscore(b)
    if az is None or bz is None:
        return -math.inf
    return float(np.mean(az * bz))


def _score_vectors(a: np.ndarray, b: np.ndarray) -> float:
    ac = _temporal_center(a)
    bc = _temporal_center(b)
    if ac.ndim != 2 or bc.ndim != 2 or ac.shape != bc.shape:
        return -math.inf
    a_std = np.std(ac, axis=0)
    b_std = np.std(bc, axis=0)
    active = (a_std > 1e-7) & (b_std > 1e-7)
    if not bool(np.any(active)):
        return -math.inf

    a_norm = ac[:, active] / a_std[active]
    b_norm = bc[:, active] / b_std[active]
    per_dimension = np.mean(a_norm * b_norm, axis=0)
    top_count = min(10, per_dimension.size)
    return float(np.mean(np.sort(per_dimension)[-top_count:]))


def _score_vectors_from_stats(stats: _VectorSearchStats, *, lag: int, overlap: int) -> float:
    src = stats.source[:overlap]
    out = stats.output[lag : lag + overlap]
    src_sum = _range_sum(stats.source_sum, 0, overlap)
    src_sq_sum = _range_sum(stats.source_sq_sum, 0, overlap)
    out_sum = _range_sum(stats.output_sum, lag, lag + overlap)
    out_sq_sum = _range_sum(stats.output_sq_sum, lag, lag + overlap)
    cross_sum = np.einsum("ij,ij->j", src, out, optimize=False)
    n = float(overlap)
    src_var_sum = src_sq_sum - (src_sum * src_sum / n)
    out_var_sum = out_sq_sum - (out_sum * out_sum / n)
    denominator = np.sqrt(np.maximum(src_var_sum, 0.0) * np.maximum(out_var_sum, 0.0))
    active = denominator > 1e-7
    if not bool(np.any(active)):
        return -math.inf

    covariance_sum = cross_sum - (src_sum * out_sum / n)
    per_dimension = covariance_sum[active] / denominator[active]
    top_count = min(10, per_dimension.size)
    return float(np.mean(np.sort(per_dimension)[-top_count:]))


def _temporal_center(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        return arr
    return arr - np.mean(arr, axis=0, keepdims=True)


def _prefix_sum(values: np.ndarray) -> np.ndarray:
    prefix = np.empty((values.shape[0] + 1, values.shape[1]), dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values, axis=0, dtype=np.float64, out=prefix[1:])
    return prefix


def _range_sum(prefix: np.ndarray, start: int, stop: int) -> np.ndarray:
    return prefix[stop] - prefix[start]


def _zscore(x: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(x, dtype=np.float32)
    std = float(np.std(arr))
    if std < 1e-7:
        return None
    return (arr - float(np.mean(arr))) / std


def _low_signal_1d(x: np.ndarray) -> bool:
    return float(np.std(x)) < 1e-6 or float(np.max(np.abs(x))) < 1e-6


def _low_signal_vectors(x: np.ndarray) -> bool:
    return float(np.std(x)) < 1e-6 or float(np.mean(np.linalg.norm(x, axis=1))) < 1e-6


def _confidence_from_score(score: float, dominance: float) -> float:
    if not math.isfinite(score):
        return 0.0
    # Correlation near 0.25 is weak but usable for noisy video fingerprints;
    # high peak dominance raises confidence when repetitive content exists.
    score_conf = (score - 0.15) / 0.65
    dominance_conf = min(1.0, dominance / 0.08)
    return max(0.0, min(1.0, 0.75 * score_conf + 0.25 * dominance_conf))
