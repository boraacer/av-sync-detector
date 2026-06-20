import numpy as np

from avsync_detector.align import estimate_delay_vectors
from avsync_detector.live import PipeHealth, audio_samples_to_features, video_frame_to_feature


def test_audio_samples_to_features_preserves_energy_events():
    samples = np.zeros(800, dtype=np.float32)
    samples[160:240] = 0.1
    samples[480:560] = 0.5

    features = audio_samples_to_features(samples, samples_per_feature=80)

    assert len(features) == 10
    assert features[2] > features[0]
    assert features[6] > features[2]


def test_pipe_health_carries_restart_and_error_details():
    health = PipeHealth(
        name="output_audio",
        running=False,
        samples=0,
        last_update_age_s=None,
        restarts=2,
        return_code=1,
        error_tail="HTTP error 404",
    )

    assert health.restarts == 2
    assert health.return_code == 1
    assert "404" in health.error_tail


def test_video_features_match_tiny_localized_motion_despite_static_texture_change():
    width = 96
    height = 54
    rate_hz = 20
    lag_frames = 46
    source = mouth_motion_frames(width=width, height=height, frames=360, lag_frames=0, seed=9)
    output = mouth_motion_frames(width=width, height=height, frames=360, lag_frames=lag_frames, seed=19, altered_static=True)

    source_features = video_features(source, width=width, height=height)
    output_features = video_features(output, width=width, height=height)
    estimate = estimate_delay_vectors(
        source_features,
        output_features,
        rate_hz=rate_hz,
        min_latency_s=0,
        max_latency_s=6,
        min_overlap_s=8,
    )

    assert abs(estimate.delay_s - (lag_frames / rate_hz)) <= 0.05
    assert estimate.confidence >= 0.6


def test_video_features_match_slow_ken_burns_pan():
    width = 96
    height = 54
    rate_hz = 20
    lag_frames = 50
    source = ken_burns_frames(width=width, height=height, frames=300, lag_frames=0)
    output = ken_burns_frames(width=width, height=height, frames=300, lag_frames=lag_frames, altered=True)

    estimate = estimate_delay_vectors(
        video_features(source, width=width, height=height),
        video_features(output, width=width, height=height),
        rate_hz=rate_hz,
        min_latency_s=0,
        max_latency_s=6,
        min_overlap_s=6,
    )

    assert abs(estimate.delay_s - (lag_frames / rate_hz)) <= 0.05
    assert estimate.confidence >= 0.5
    assert "max_latency_boundary" not in estimate.warnings


def test_static_back_of_head_style_video_stays_low_confidence():
    width = 96
    height = 54
    rate_hz = 20
    lag_frames = 50
    source = back_of_head_frames(width=width, height=height, frames=300, lag_frames=0, seed=31)
    output = back_of_head_frames(width=width, height=height, frames=300, lag_frames=lag_frames, seed=43, altered=True)

    estimate = estimate_delay_vectors(
        video_features(source, width=width, height=height),
        video_features(output, width=width, height=height),
        rate_hz=rate_hz,
        min_latency_s=0,
        max_latency_s=6,
        min_overlap_s=6,
    )

    assert estimate.confidence < 0.2
    assert "low_confidence" in estimate.warnings


def video_features(frames: np.ndarray, *, width: int, height: int) -> np.ndarray:
    features: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for frame in frames:
        feature, previous = video_frame_to_feature(frame, previous=previous, width=width, height=height)
        features.append(feature)
    return np.asarray(features, dtype=np.float32)


def mouth_motion_frames(
    *,
    width: int,
    height: int,
    frames: int,
    lag_frames: int,
    seed: int,
    altered_static: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width), dtype=np.float32)
    base[:, :] = np.linspace(40, 170, width, dtype=np.float32)[None, :]
    for y in range(height):
        for x in range(width):
            if ((x - 48) / 18) ** 2 + ((y - 26) / 22) ** 2 < 1:
                base[y, x] = 145 + ((x * 3 + y * 5) % 17)
    if altered_static:
        base = base * 0.86 + 18
        base += rng.normal(0, 4.5, base.shape)

    out = []
    for i in range(frames):
        source_i = max(0, i - lag_frames)
        frame = base.copy()
        amp = 12 * np.sin(source_i * 0.31) + 7 * np.sin(source_i * 0.83) + 4 * np.sin(source_i * 1.37)
        frame[33:36, 43:54] += amp * (0.8 if altered_static else 1.0)
        frame += rng.normal(0, 2.2 if altered_static else 1.2, frame.shape)
        out.append(np.clip(frame, 0, 255).reshape(-1))
    return np.asarray(out, dtype=np.float32)


def ken_burns_frames(*, width: int, height: int, frames: int, lag_frames: int, altered: bool = False) -> np.ndarray:
    base = textured_frame(width=width, height=height, seed=4)
    rng = np.random.default_rng(9 if altered else 5)
    out = []
    for i in range(frames):
        source_i = max(0, i - lag_frames)
        frame = shifted_frame(
            base,
            dx=0.025 * source_i + 0.5 * np.sin(source_i * 0.035),
            dy=0.010 * source_i + 0.35 * np.sin(source_i * 0.027),
        )
        if altered:
            frame = frame * 0.93 + 8 + rng.normal(0, 1.1, frame.shape)
        out.append(np.clip(frame, 0, 255).reshape(-1))
    return np.asarray(out, dtype=np.float32)


def back_of_head_frames(
    *,
    width: int,
    height: int,
    frames: int,
    lag_frames: int,
    seed: int,
    altered: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width), dtype=np.float32) + 80
    for y in range(height):
        for x in range(width):
            if ((x - width / 2) / 19) ** 2 + ((y - height * 0.52) / 24) ** 2 < 1:
                base[y, x] = 105 + ((x * 7 + y * 3) % 11)

    out = []
    for i in range(frames):
        source_i = max(0, i - lag_frames)
        frame = base.copy()
        frame += 0.35 * np.sin(source_i * 0.04) + rng.normal(0, 0.65 if altered else 0.45, frame.shape)
        if altered:
            frame = frame * 0.98 + 2
        out.append(np.clip(frame, 0, 255).reshape(-1))
    return np.asarray(out, dtype=np.float32)


def textured_frame(*, width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(110, 25, (height, width)).astype(np.float32)
    for _ in range(3):
        base = (base + np.roll(base, 1, 0) + np.roll(base, -1, 0) + np.roll(base, 1, 1) + np.roll(base, -1, 1)) / 5
    return np.clip(base, 0, 255)


def shifted_frame(base: np.ndarray, *, dx: float, dy: float) -> np.ndarray:
    height, width = base.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    x = xx - dx
    y = yy - dy
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = x - x0
    wy = y - y0
    fill = float(np.mean(base))

    def sample(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        ok = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        values = np.full((height, width), fill, dtype=np.float32)
        values[ok] = base[iy[ok], ix[ok]]
        return values

    return (
        sample(x0, y0) * (1 - wx) * (1 - wy)
        + sample(x1, y0) * wx * (1 - wy)
        + sample(x0, y1) * (1 - wx) * wy
        + sample(x1, y1) * wx * wy
    )
