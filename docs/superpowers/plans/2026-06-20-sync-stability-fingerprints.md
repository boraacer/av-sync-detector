# Sync Stability Fingerprints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce volatile A/V offset reporting by requiring repeated agreement between recent estimates and by improving video matching for slow pan/Ken Burns footage without trusting static shots.

**Architecture:** Keep the existing source-vs-output latency estimator, but make concrete offset publication stateful in `LiveAnalyzer`. Improve vector scoring so video fingerprints reward temporal variation rather than static appearance that matches at every lag.

**Tech Stack:** Python 3.11, NumPy, pytest, Rich TUI.

---

### Task 1: Stability Gate

**Files:**
- Modify: `src/avsync_detector/live.py`
- Test: `tests/test_live_analyzer.py`

- [ ] **Step 1: Write failing tests**

Add tests that feed `LiveAnalyzer.estimate()` a sequence of alternating concrete offsets and assert that the analyzer returns `inconclusive` with `unstable_offset`, then feed agreeing offsets and assert that the concrete offset is published.

- [ ] **Step 2: Verify red**

Run: `pytest -q tests/test_live_analyzer.py::test_live_analyzer_hides_volatile_offsets_until_recent_estimates_agree tests/test_live_analyzer.py::test_live_analyzer_publishes_offset_after_recent_estimates_agree`

Expected: tests fail because `LiveAnalyzer` currently publishes every concrete estimate immediately.

- [ ] **Step 3: Implement minimal gate**

Add stability options to `LiveOptions`, keep a recent concrete-estimate buffer in `LiveAnalyzer`, and replace volatile concrete estimates with an inconclusive result that preserves latencies/confidence and adds `unstable_offset` plus `unreliable_offset`.

- [ ] **Step 4: Verify green**

Run: `pytest -q tests/test_live_analyzer.py::test_live_analyzer_hides_volatile_offsets_until_recent_estimates_agree tests/test_live_analyzer.py::test_live_analyzer_publishes_offset_after_recent_estimates_agree`

Expected: tests pass.

### Task 2: Video Scoring

**Files:**
- Modify: `src/avsync_detector/align.py`
- Modify: `src/avsync_detector/live.py`
- Test: `tests/test_live_features.py`

- [ ] **Step 1: Write failing tests**

Add a slow Ken Burns/pan synthetic frame test that currently hits a search boundary or low confidence, and add a static back-of-head-style test that must remain low confidence.

- [ ] **Step 2: Verify red**

Run: `pytest -q tests/test_live_features.py::test_video_features_match_slow_ken_burns_pan tests/test_live_features.py::test_static_back_of_head_style_video_stays_low_confidence`

Expected: slow-pan test fails under the existing motion-only fingerprint or scoring.

- [ ] **Step 3: Implement minimal scoring/features**

Extend per-frame video features with block-level appearance gradient statistics, and update vector scoring to remove per-feature means over the overlap before correlation. This keeps temporal changes while suppressing static scene identity.

- [ ] **Step 4: Verify green**

Run: `pytest -q tests/test_live_features.py::test_video_features_match_slow_ken_burns_pan tests/test_live_features.py::test_static_back_of_head_style_video_stays_low_confidence`

Expected: tests pass.

### Task 3: Docs and Regression

**Files:**
- Modify: `docs/algorithm.md`
- Test: full suite

- [ ] **Step 1: Update docs**

Document estimate stability gating and temporal-centered video fingerprints in `docs/algorithm.md`.

- [ ] **Step 2: Run focused tests**

Run: `pytest -q tests/test_align.py tests/test_live_features.py tests/test_live_analyzer.py`

Expected: all focused tests pass.

- [ ] **Step 3: Run full suite**

Run: `pytest -q`

Expected: all tests pass.
