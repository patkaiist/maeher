from __future__ import annotations

import numpy as np

from maeher import _core

# REAPER's Init() needs enough signal energy for its RMS normalisation and
# enough samples for its LPC/NCCF windows.  Below these thresholds we skip
# the C++ call entirely and return all-zero / empty arrays.
_MIN_RMS = 1e-6
_MIN_DURATION_S = 0.1  # ~100 ms; anything shorter can't fit a full analysis window


def _empty_result(n_samples: int, sample_rate: int, frame_interval: float) -> dict[str, np.ndarray]:
    duration = n_samples / sample_rate
    n_frames = max(0, int(duration / frame_interval))
    return {
        "time":         np.arange(n_frames, dtype=np.float32) * frame_interval,
        "f0":           np.zeros(n_frames, dtype=np.float32),
        "voiced":       np.zeros(n_frames, dtype=bool),
        "corr":         np.zeros(n_frames, dtype=np.float32),
        "epochs":       np.empty(0, dtype=np.float32),
        "epoch_voiced": np.empty(0, dtype=bool),
    }


def track(
    audio: np.ndarray,
    *,
    sample_rate: int = 16000,
    min_f0: float = 40.0,
    max_f0: float = 500.0,
    frame_interval: float = 0.005,
    unvoiced_pulse_interval: float = 0.01,
    unvoiced_cost: float = 0.9,
    do_highpass: bool = True,
    do_hilbert_transform: bool = False,
) -> dict[str, np.ndarray]:
    """Estimate F0 from a mono audio signal using REAPER.

    Parameters
    ----------
    audio:
        1-D NumPy array of mono audio samples.  float32 preferred; any
        floating-point dtype is accepted and converted without a copy when
        possible.  Expected amplitude range is [-1, 1].
    sample_rate:
        Sample rate of *audio* in Hz.
    min_f0:
        Minimum F0 to search for, in Hz.
    max_f0:
        Maximum F0 to search for, in Hz.
    frame_interval:
        Output frame period in seconds.
    unvoiced_pulse_interval:
        Spacing of synthetic pitch marks inserted in unvoiced regions.
    unvoiced_cost:
        DP cost for the unvoiced hypothesis.  Higher values bias towards
        more voiced frames in noisy conditions.
    do_highpass:
        Apply 80 Hz highpass filter to remove low-frequency rumble.
    do_hilbert_transform:
        Apply Hilbert transform to reduce phase distortion (useful for
        close-talking microphone recordings).

    Returns
    -------
    dict with keys:

    ``time``
        float32 array — timestamp of each F0 frame (seconds).
    ``f0``
        float32 array — estimated F0 in Hz; 0 in unvoiced frames.
    ``voiced``
        bool array — ``True`` where voicing was detected.
    ``corr``
        float32 array — normalized cross-correlation value per frame.
    ``epochs``
        float32 array — glottal closure instant times (seconds).
    ``epoch_voiced``
        bool array — voicing state at each epoch.

    Notes
    -----
    Near-silence (RMS < 1e-6) and very short signals (< 100 ms) cannot be
    analysed by the upstream REAPER algorithm.  In those cases a zero-filled
    result is returned rather than raising an exception.
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1-D, got shape {audio.shape}")

    n = len(audio)
    duration = n / sample_rate

    if duration < _MIN_DURATION_S or float(np.sqrt(np.mean(audio ** 2))) < _MIN_RMS:
        return _empty_result(n, sample_rate, frame_interval)

    return _core.track(
        audio,
        sample_rate=sample_rate,
        min_f0=min_f0,
        max_f0=max_f0,
        frame_interval=frame_interval,
        unvoiced_pulse_interval=unvoiced_pulse_interval,
        unvoiced_cost=unvoiced_cost,
        do_highpass=do_highpass,
        do_hilbert_transform=do_hilbert_transform,
    )
