# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Unit tests for on-device pure NumPy acoustic voice adaptor (SCRUM-533).
"""
import time
import numpy as np
import pytest

from termux_tts.acoustic_adaptor import (
    pure_numpy_phase_vocoder,
    pure_numpy_pitch_shift,
    pure_numpy_spectral_shaping,
    adapt_waveform_to_profile,
)
from termux_tts.speaker_profile import SpeakerProfile, EMBEDDING_DIM


def test_phase_vocoder_duration():
    """Verify phase vocoder time-stretch ratio."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Speed up by 2x -> duration should halve
    fast = pure_numpy_phase_vocoder(x, speed=2.0)
    assert not np.isnan(fast).any()
    assert not np.isinf(fast).any()
    expected_samples = sr // 2
    assert abs(len(fast) - expected_samples) < sr * 0.1

    # Slow down by 0.5x -> duration should double
    slow = pure_numpy_phase_vocoder(x, speed=0.5)
    assert not np.isnan(slow).any()
    assert abs(len(slow) - sr * 2) < sr * 0.1


def test_pitch_shift_frequency_shift():
    """Verify that pure numpy pitch shift transposes frequency without changing duration."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f_in = 300.0
    x = np.sin(2 * np.pi * f_in * t).astype(np.float32)

    # Shift down by 1 octave (-12 semitones) -> should be ~150Hz
    t0 = time.perf_counter()
    shifted_down = pure_numpy_pitch_shift(x, semitones=-12.0, sr=sr)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(shifted_down) == len(x)
    assert not np.isnan(shifted_down).any()
    assert elapsed_ms < 150.0  # Fast execution

    # Autocorrelation to check fundamental frequency
    corr = np.correlate(shifted_down, shifted_down, mode="full")
    corr = corr[len(corr)//2:]
    min_lag = int(sr / 350.0)
    max_lag = int(sr / 100.0)
    lag = min_lag + np.argmax(corr[min_lag:max_lag])
    f_out = sr / lag
    # Expected: 150Hz (+/- 15Hz tolerance for finite window)
    assert 135.0 <= f_out <= 165.0


def test_adapt_waveform_to_profile_end_to_end():
    """Verify full adaptation against a masculine target SpeakerProfile."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # High pitch female tone (300Hz)
    x = (0.6 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)

    fake_emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    profile = SpeakerProfile(
        name="syuka_target",
        embedding=fake_emb,
        sample_rate=16000,
        duration_sec=60.0,
        created_at="2026-10-11T20:00:00Z",
        pitch_mean_hz=105.0,  # Masculine target pitch
        energy_rms_db=-18.0,
    )

    t0 = time.perf_counter()
    adapted = adapt_waveform_to_profile(x, sr, profile, source_f0_estimate=300.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(adapted) == len(x)
    assert not np.isnan(adapted).any()
    assert np.max(np.abs(adapted)) <= 1.0
    print(f"Full waveform adaptation elapsed: {elapsed_ms:.2f}ms for 2.0s audio")
    assert elapsed_ms < 100.0  # Must be sub-100ms
