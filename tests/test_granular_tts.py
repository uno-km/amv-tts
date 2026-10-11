# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Production-Grade Invariant & Acoustic Safety Test Suite for termux-tts.
Validates G2P correctness, Number Normalization, Acoustic Signal Energy, Filter Stability, and Security Bounds.
"""

import os
import time
import pytest
import numpy as np

import termux_tts as tts
from termux_tts.tokenizer import PhoneticTokenizer, decompose_hangul
from termux_tts.audio import AudioBuffer
from termux_tts.engine import TTSEngine, load
from termux_tts.exceptions import (
    TTSError,
    TTSModelLoadError,
    TTSInferenceError,
    TTSAudioEncodingError,
    TTSLanguageNotSupportedError,
)

# ==============================================================================
# 1. Phonetic Tokenization & Number Normalization Invariants
# ==============================================================================

def test_hangul_jamo_decomposition():
    jamos = decompose_hangul("한")
    assert jamos == ["ㅎ", "ㅏ", "ㄴ"]
    jamos_no_jong = decompose_hangul("가")
    assert jamos_no_jong == ["ㄱ", "ㅏ"]

def test_korean_number_normalization():
    from termux_tts.tokenizer import normalize_numbers_korean
    tok = PhoneticTokenizer(language="ko")
    # 1. Place-value Sino-Korean string conversions
    assert normalize_numbers_korean("100") == "백"
    assert normalize_numbers_korean("1234") == "천이백삼십사"
    assert normalize_numbers_korean("10000") == "만"
    assert normalize_numbers_korean("2026년 9월 1일") == "이천이십육년 구월 일일"

    # 2. G2P-applied phonetic normalization
    phonetic_1234 = tok.normalize_text("1234")
    assert "처니" in phonetic_1234  # Liaison: 천+이 -> 처니
    assert "백" in phonetic_1234
    
    tokens = tok.tokenize("2026년 9월 1일")
    assert len(tokens) > 10
    assert all(isinstance(t, int) for t in tokens)

def test_english_number_and_word_tokenization():
    tok = PhoneticTokenizer(language="en")
    norm_text = tok.normalize_text("System 123 active")
    assert "one" in norm_text
    assert "two" in norm_text
    assert "three" in norm_text
    tokens = tok.tokenize("System 123 active")
    assert len(tokens) > 10

def test_unsupported_language_guard():
    with pytest.raises(TTSLanguageNotSupportedError):
        PhoneticTokenizer(language="fr_unsupported")

# ==============================================================================
# 2. Audio Buffer & Acoustic Signal Invariants
# ==============================================================================

def test_audio_buffer_normalization():
    raw = np.array([2.5, -3.0, 1.0, 0.5], dtype=np.float32)
    buf = AudioBuffer(raw, sample_rate=22050)
    assert np.max(np.abs(buf.samples)) <= 1.0
    assert not np.isnan(buf.samples).any()
    assert not np.isinf(buf.samples).any()

def test_riff_wav_byte_encoding():
    raw = np.sin(np.linspace(0, 2 * np.pi * 440, 2205, dtype=np.float32))
    buf = AudioBuffer(raw, sample_rate=22050)
    wav_bytes = buf.to_wav_bytes()
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes
    assert len(wav_bytes) > 2205 * 2

def test_audio_buffer_dynamic_range():
    """Verify AudioBuffer does not explode or produce NaNs."""
    impulse = np.zeros(1000, dtype=np.float32)
    impulse[0] = 1.0
    buf = AudioBuffer(impulse, sample_rate=22050)
    assert not np.isnan(buf.samples).any()
    assert not np.isinf(buf.samples).any()
    assert np.max(np.abs(buf.samples)) <= 1.0

# ==============================================================================
# 3. Acoustic Synthesis & Fail-Fast Integrity
# ==============================================================================

def test_neural_synthesis_korean_acoustics():
    try:
        engine = load(language="ko")
    except (TTSModelLoadError, TTSInferenceError):
        pytest.skip("Neural model not available in local test environment")
    with engine:
        res = engine.synthesize("안녕하세요, 텀묵스 음향 합성 무결성 검증입니다.")
        assert res.duration_sec > 0.5
        assert res.sample_rate == 22050
        assert res.rtf < 0.5
        assert len(res.wav_bytes) > 1000
        # Acoustic energy check: verify non-silent valid speech signal
        rms = np.sqrt(np.mean(res.audio_buffer.samples ** 2))
        assert rms > 0.005, f"Audio signal is virtually silent (RMS={rms})"

def test_neural_synthesis_english_acoustics():
    try:
        engine = load(language="en")
    except (TTSModelLoadError, TTSInferenceError):
        pytest.skip("Neural model not available in local test environment")
    with engine:
        res = engine.synthesize("Hello world, this is termux speech synthesis.")
        assert res.duration_sec > 0.5
        rms = np.sqrt(np.mean(res.audio_buffer.samples ** 2))
        assert rms > 0.005

def test_zero_fallback_error_guards():
    try:
        engine = load(language="ko")
    except (TTSModelLoadError, TTSInferenceError):
        pytest.skip("Neural model not available in local test environment")
    with engine:
        with pytest.raises(TTSInferenceError):
            engine.synthesize("")
        with pytest.raises(TTSInferenceError):
            engine.synthesize("   ")
        with pytest.raises(TTSInferenceError):
            engine.synthesize("Hello", speed=-1.0)
        with pytest.raises(TTSInferenceError):
            engine.synthesize("Hello", speed=5.0)

# ==============================================================================
# 4. Lifecycle & Performance Scaling
# ==============================================================================

def test_raii_context_manager_lifecycle():
    try:
        engine = load(language="ko")
    except (TTSModelLoadError, TTSInferenceError):
        pytest.skip("Neural model not available in local test environment")
    with engine:
        assert not engine._is_closed
    assert engine._is_closed
    with pytest.raises(TTSInferenceError):
        engine.synthesize("Valid text")

def test_speed_scaling_contract():
    try:
        engine = load(language="ko")
    except (TTSModelLoadError, TTSInferenceError):
        pytest.skip("Neural model not available in local test environment")
    with engine:
        res_normal = engine.synthesize("속도 테스트 문장입니다.", speed=1.0)
        res_fast = engine.synthesize("속도 테스트 문장입니다.", speed=2.0)
        assert res_fast.duration_sec < res_normal.duration_sec
