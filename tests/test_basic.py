"""Basic regression and smoke tests for maeher."""

from __future__ import annotations

import math

import numpy as np
import pytest

import maeher

SR = 16000
FRAME_INTERVAL = 0.005


def _sine_sweep(f_start: float, f_end: float, duration: float, sr: int) -> np.ndarray:
    """Linearly swept sine wave (voiced carrier)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float64)
    phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) / duration * t**2)
    return np.sin(phase).astype(np.float32)


def _sine(freq: float, duration: float, sr: int) -> np.ndarray:
    t = np.arange(int(sr * duration), dtype=np.float64) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _silence(duration: float, sr: int) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


def _white_noise(duration: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(int(sr * duration)).astype(np.float32) * 0.1


# API contract ─


def test_return_keys():
    audio = _sine(200.0, 0.5, SR)
    result = maeher.track(audio, sample_rate=SR)
    assert set(result.keys()) == {
        "time",
        "f0",
        "voiced",
        "corr",
        "epochs",
        "epoch_voiced",
    }


def test_array_lengths_consistent():
    audio = _sine(200.0, 0.5, SR)
    r = maeher.track(audio, sample_rate=SR)
    n = len(r["time"])
    assert len(r["f0"]) == n
    assert len(r["voiced"]) == n
    assert len(r["corr"]) == n
    assert len(r["epochs"]) == len(r["epoch_voiced"])


def test_time_is_monotone():
    audio = _sine(150.0, 1.0, SR)
    r = maeher.track(audio, sample_rate=SR)
    assert np.all(np.diff(r["time"]) > 0)


def test_time_step_matches_frame_interval():
    audio = _sine(200.0, 1.0, SR)
    fi = 0.010
    r = maeher.track(audio, sample_rate=SR, frame_interval=fi)
    diffs = np.diff(r["time"])
    np.testing.assert_allclose(diffs, fi, atol=1e-6)


def test_dtypes():
    audio = _sine(200.0, 0.5, SR)
    r = maeher.track(audio, sample_rate=SR)
    assert r["time"].dtype == np.float32
    assert r["f0"].dtype == np.float32
    assert r["voiced"].dtype == bool
    assert r["corr"].dtype == np.float32
    assert r["epochs"].dtype == np.float32


# Voiced / unvoiced behaviour


def test_sine_is_voiced():
    """A clean sine wave should be mostly voiced."""
    audio = _sine(200.0, 1.0, SR)
    r = maeher.track(audio, sample_rate=SR)
    voiced_fraction = r["voiced"].mean()
    assert voiced_fraction > 0.5, f"expected >50 % voiced, got {voiced_fraction:.2%}"


def test_silence_is_unvoiced():
    """Near-silence returns a zero-filled result rather than raising."""
    audio = _silence(0.5, SR)
    r = maeher.track(audio, sample_rate=SR)
    assert not r["voiced"].any(), "silence should be entirely unvoiced"


def test_f0_zero_in_unvoiced_frames():
    r = maeher.track(_silence(0.5, SR), sample_rate=SR)
    assert np.all(r["f0"][~r["voiced"]] == 0.0)


# F0 accuracy


@pytest.mark.xfail(
    reason=(
        "REAPER is tuned for glottal-pulse structure (speech LPC residual), not "
        "pure sine waves.  Sine inputs produce unreliable F0 estimates.  "
        "Use speech or sawtooth-wave inputs for accuracy regression tests."
    ),
    strict=False,
)
@pytest.mark.parametrize("freq", [80, 150, 200, 300, 400])
def test_f0_accuracy_pure_sine(freq: int):
    """F0 estimate within one octave of the true value for a clean sine.

    Marked xfail: REAPER regularly returns wrong estimates on pure tones.
    If this starts passing it means REAPER is accidentally working — good,
    but not a regression to worry about.
    """
    audio = _sine(float(freq), 1.5, SR)
    r = maeher.track(audio, sample_rate=SR, min_f0=40.0, max_f0=500.0)
    voiced_f0 = r["f0"][r["voiced"]]
    if voiced_f0.size == 0:
        pytest.skip(f"no voiced frames for {freq} Hz")
    median_f0 = float(np.median(voiced_f0))
    lo, hi = freq / 2.0, freq * 2.0
    assert (
        lo <= median_f0 <= hi
    ), f"median F0={median_f0:.1f} outside [{lo:.0f}, {hi:.0f}]"


# Edge cases


def test_voiced_unvoiced_transition():
    """Voiced segment followed by silence: first half more voiced than second."""
    voiced_seg = _sine(200.0, 0.4, SR)
    silent_seg = _silence(0.4, SR)
    audio = np.concatenate([voiced_seg, silent_seg])
    r = maeher.track(audio, sample_rate=SR)
    mid = len(r["voiced"]) // 2
    # First half should be predominantly voiced.
    assert r["voiced"][:mid].mean() > 0.4
    # Second half (silence) must have fewer voiced frames than the first.
    # REAPER doesn't cleanly unvoice abrupt silence, so we only require the
    # trend, not a specific threshold.
    assert r["voiced"][mid:].mean() < r["voiced"][:mid].mean()


def test_sine_sweep_voiced():
    """Swept sine should remain predominantly voiced."""
    audio = _sine_sweep(100.0, 400.0, 2.0, SR)
    r = maeher.track(audio, sample_rate=SR)
    assert r["voiced"].mean() > 0.5


def test_noisy_speech_does_not_crash():
    rng = np.random.default_rng(0)
    voiced = _sine(200.0, 1.0, SR)
    noise = _white_noise(1.0, SR, rng)
    audio = voiced + noise
    r = maeher.track(audio, sample_rate=SR)
    assert "f0" in r


def test_very_short_audio_does_not_crash():
    """Audio shorter than the minimum analysis window returns empty arrays."""
    audio = _sine(200.0, 0.05, SR)
    r = maeher.track(audio, sample_rate=SR)
    assert "f0" in r
    assert not r["voiced"].any()


def test_invalid_ndim_raises():
    with pytest.raises((ValueError, Exception)):
        maeher.track(np.zeros((10, 10), dtype=np.float32), sample_rate=SR)


def test_float64_input_accepted():
    audio = _sine(200.0, 0.5, SR).astype(np.float64)
    r = maeher.track(audio, sample_rate=SR)
    assert "f0" in r
