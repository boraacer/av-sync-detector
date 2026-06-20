# Algorithm Notes

AV Sync Detector compares a source URL with an output URL by matching content-derived fingerprints. It deliberately avoids stream timestamps because bad inputs, transcoders, RTMP, and HLS packagers can rewrite or corrupt timing metadata.

## Capture Model

The detector starts one FFmpeg process per URL. Each FFmpeg process extracts audio and video from the same input session:

- audio is written as mono `f32le`
- video is written as grayscale raw frames

Using one input session per URL matters. If audio and video are captured by separate FFmpeg processes, live HLS or HTTP inputs can attach at different live positions and create a fake A/V offset.

## Audio Fingerprints

Audio is converted to compact energy features:

- mono downmix
- fixed sample rate
- short feature frames
- RMS and peak energy
- log compression

These features are intentionally lightweight and robust to transcoding. They work well for speech and program audio with enough energy variation, but repeated music or steady ambience can be ambiguous.

## Video Fingerprints

Video is converted to localized block-motion fingerprints:

- downscale to a small grayscale frame
- compare each frame with the previous frame
- divide the frame into a grid
- record block-level motion statistics
- robustly normalize each feature vector

This is designed for live broadcast checks, especially static talking-head shots where the useful signal may be a small moving mouth or face area.

## Delay Search

For audio and video independently, the detector searches for the source-to-output delay that maximizes normalized correlation between source and output fingerprints.

The final sync offset is:

```text
av_offset_ms = (video_latency_s - audio_latency_s) * 1000
```

Interpretation:

- positive offset: audio is ahead of video
- negative offset: video is ahead of audio
- near zero: audio and video are aligned

Large source-to-output latency is acceptable when audio and video share the same delay.

## Confidence

The detector reports `inconclusive` rather than a concrete A/V offset when evidence is weak. Low-confidence latency guesses are retained as diagnostics, but the final offset remains unknown.

Boundary matches are treated as unreliable. If the best match occurs at the minimum or maximum searched latency, the search probably clipped the true match or saw a coincidental pattern. In that case, increase `--duration`, `--window`, or `--max-latency`.

## Practical Settings

For HLS distribution outputs, start with:

```bash
avsync-detector json \
  --source "<source-url>" \
  --output "<output-url>" \
  --duration 70 \
  --window 120 \
  --max-latency 60
```

Use longer durations when:

- output HLS is far behind the source
- confidence remains low
- warnings include `min_latency_boundary` or `max_latency_boundary`
- the video is mostly static
- content is repetitive

## Limitations

The detector is a measurement tool, not a proof engine. It can still be inconclusive on:

- repeated music or repeated visual loops
- long static scenes with no meaningful motion
- streams with missing audio or video
- streams whose source/output content differs materially
- insufficient observation duration

When the result is inconclusive, do not treat diagnostic latency guesses as a sync verdict.

