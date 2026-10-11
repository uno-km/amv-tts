# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Unit tests for SpeakerProfile and SpeakerProfileEngine (SCRUM-528).
"""
import tempfile
import time
from pathlib import Path
import numpy as np
import pytest

from termux_tts.speaker_profile import (
    SpeakerProfile,
    SpeakerProfileEngine,
    SpeakerProfileError,
    ProfileNotFoundError,
    EMBEDDING_DIM,
)
from termux_tts.audio import AudioBuffer


def create_synthetic_wav(filepath: Path, duration_sec: float = 3.0, freq: float = 220.0, sr: int = 16000):
    """Generate a clean synthetic sine tone speech dummy WAV."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Fundamental frequency + harmonics to simulate vocal cords
    signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * (freq * 2) * t) + 0.1 * np.sin(2 * np.pi * (freq * 3) * t)
    buf = AudioBuffer(signal, sample_rate=sr)
    buf.save(str(filepath))
    return filepath


def test_speaker_profile_unit_norm():
    """Verify that any embedding is strictly L2-normalized to 1.0."""
    raw = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    prof = SpeakerProfile(
        name="test_norm",
        embedding=raw,
        sample_rate=16000,
        duration_sec=3.5,
        created_at="2026-10-11T00:00:00Z",
        pitch_mean_hz=180.0,
        energy_rms_db=-18.0,
    )
    norm = np.linalg.norm(prof.embedding)
    assert abs(norm - 1.0) < 1e-5
    assert len(prof.embedding) == 192


def test_profile_serialization_roundtrip():
    """Verify binary .vec saving, loading, and metadata parity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        prof = SpeakerProfile(
            name="alice",
            embedding=raw,
            sample_rate=16000,
            duration_sec=4.2,
            created_at="2026-10-11T12:00:00Z",
            pitch_mean_hz=210.5,
            energy_rms_db=-14.2,
            source_file="alice_raw.wav",
        )
        saved_vec = prof.save(directory=tmp_path)
        assert saved_vec.is_file()
        assert (tmp_path / "alice.meta.json").is_file()

        loaded = SpeakerProfile.load("alice", directory=tmp_path)
        assert loaded.name == "alice"
        assert abs(loaded.pitch_mean_hz - 210.5) < 1e-4
        assert abs(loaded.duration_sec - 4.2) < 1e-4
        assert np.allclose(prof.embedding, loaded.embedding, atol=1e-5)


def test_speaker_profile_engine_extract_and_similarity():
    """Test feature extraction from synthetic WAV and cosine similarity metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        wav1 = create_synthetic_wav(tmp_path / "speaker_a1.wav", duration_sec=3.0, freq=140.0)
        wav2 = create_synthetic_wav(tmp_path / "speaker_a2.wav", duration_sec=3.0, freq=142.0)
        wav3 = create_synthetic_wav(tmp_path / "speaker_b.wav", duration_sec=3.0, freq=300.0)

        engine = SpeakerProfileEngine(profiles_dir=tmp_path / "profiles")
        t0 = time.perf_counter()
        prof_a1 = engine.extract_from_file(wav1, name="speaker_a1")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 50.0, f"Extraction must be < 50ms, took {elapsed_ms:.2f}ms"

        prof_a2 = engine.extract_from_file(wav2, name="speaker_a2")
        prof_b = engine.extract_from_file(wav3, name="speaker_b")

        # Save and list
        prof_a1.save(tmp_path / "profiles")
        prof_a2.save(tmp_path / "profiles")
        all_profs = engine.list_profiles()
        assert len(all_profs) == 2

        # Check self-similarity == 1.0
        assert abs(SpeakerProfileEngine.compute_similarity(prof_a1, prof_a1) - 1.0) < 1e-5

        # Check intra-speaker similarity (A1 vs A2) > inter-speaker similarity (A1 vs B)
        sim_same = SpeakerProfileEngine.compute_similarity(prof_a1, prof_a2)
        sim_diff = SpeakerProfileEngine.compute_similarity(prof_a1, prof_b)
        assert sim_same > sim_diff, f"Expected {sim_same} > {sim_diff}"
