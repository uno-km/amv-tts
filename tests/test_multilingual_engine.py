# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Unit & Zero-Deception Test Suite for Extensible Multilingual Neural Engine and TTSEngine Gateway.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from termux_tts.audio import AudioBuffer
from termux_tts.script_classifier import MultilingualTokenizer, LanguageChunk
from termux_tts.engine_multilingual import (
    MultilingualNeuralEngine,
    MultilingualResult,
    trim_silence_samples,
)
from termux_tts.engine import TTSEngine
from termux_tts.exceptions import TTSModelLoadError, TTSInferenceError


class MockSubEngine:
    """Mock engine that returns deterministic sine wave AudioBuffer."""
    def __init__(self, lang_label: str, sample_rate: int = 22050):
        self.lang_label = lang_label
        self.sample_rate = sample_rate

    def synthesize(self, text: str, speed: float = 1.0, **kwargs):
        # 0.2s of audio with leading and trailing silence
        silence = np.zeros(200, dtype=np.float32)
        signal = np.sin(np.linspace(0, 2 * np.pi * 440 * 0.1, 2205)).astype(np.float32)
        samples = np.concatenate([silence, signal, silence])
        buf = AudioBuffer(samples, sample_rate=self.sample_rate)
        
        mock_res = MagicMock()
        mock_res.audio_buffer = buf
        mock_res.sample_rate = self.sample_rate
        mock_res.duration_sec = buf.duration_seconds
        mock_res.rtf = 0.05
        return mock_res


def test_trim_silence_samples():
    samples = np.array([0.0, 0.001, 0.05, 0.8, -0.5, 0.002, 0.0], dtype=np.float32)
    trimmed = trim_silence_samples(samples, threshold=0.01)
    assert len(trimmed) == 3
    assert trimmed[0] == pytest.approx(0.05)
    assert trimmed[-1] == pytest.approx(-0.5)


def test_multilingual_engine_mock_synthesis():
    engine = MultilingualNeuralEngine(sample_rate=22050)
    
    # Register mock engines for ko and en
    mock_ko = MockSubEngine("ko", sample_rate=22050)
    mock_en = MockSubEngine("en", sample_rate=22050)
    engine.register_engine("ko", mock_ko)
    engine.register_engine("en", mock_en)

    text = "Hello 반가워 나는 parrot이라고 해"
    result = engine.synthesize(text, mode="stitch")

    assert isinstance(result, MultilingualResult)
    assert "en" in result.languages_detected
    assert "ko" in result.languages_detected
    assert result.duration_sec > 0.0
    assert result.sample_rate == 22050
    assert len(result.chunks) >= 3


def test_multilingual_engine_unified_mode():
    engine = MultilingualNeuralEngine(sample_rate=22050)
    mock_ko = MockSubEngine("ko", sample_rate=22050)
    engine.register_engine("ko", mock_ko)

    text = "Hello 반가워 나는 parrot이라고 해"
    result = engine.synthesize(text, mode="unified")

    assert isinstance(result, MultilingualResult)
    assert result.backend == "UNIFIED_SINGLE_PASS_NEURAL"
    assert "ko" in result.languages_detected
    assert len(result.chunks) == 1
    assert result.duration_sec > 0.0


def test_zero_silent_fallback_on_unprovisioned_language():
    engine = MultilingualNeuralEngine(sample_rate=22050)
    mock_ko = MockSubEngine("ko", sample_rate=22050)
    engine.register_engine("ko", mock_ko)
    # English engine is NOT registered and not installed

    with pytest.raises(TTSModelLoadError) as exc_info:
        # In stitch mode, unprovisioned English chunk fails fast
        engine.synthesize("Hello 반갑습니다", mode="stitch")
    
    assert "[FAIL-FAST]" in str(exc_info.value)


def test_tts_engine_gateway_auto_routing():
    # Verify TTSEngine routes to MultilingualNeuralEngine in auto mode for all speech
    with patch.object(TTSEngine, "_resolve_synth_engine", return_value=MagicMock()):
        tts = TTSEngine(engine_type="auto", language="auto")
        
        # Mock multilingual engine
        mock_multi = MagicMock()
        mock_multi.synthesize.return_value = MagicMock(spec=MultilingualResult)
        tts._multilingual_engine = mock_multi

        # 1. In auto mode, text routes to multilingual orchestrator by default
        tts.synthesize("Hello 안녕하세요")
        mock_multi.synthesize.assert_called_once_with("Hello 안녕하세요", output=None, speed=1.0, mode="unified")

    # 2. When explicit native engine is requested, routes to native engine speak
    tts_native = TTSEngine(engine_type="native")
    with patch.object(tts_native.native_engine, "speak", return_value=MagicMock()) as mock_speak:
        tts_native.synthesize("Native speech text")
        mock_speak.assert_called_once_with("Native speech text")


def test_resident_manager_lifecycle():
    from termux_tts.engine_sherpa_capi import SherpaResidentManager
    mgr = SherpaResidentManager.get_instance()
    assert mgr is not None
    # Verify singleton
    mgr2 = SherpaResidentManager.get_instance()
    assert mgr is mgr2

    # Unprovisioned language resolution fails fast
    with pytest.raises(TTSModelLoadError) as exc:
        mgr.resolve_model_assets("non_existent_lang")
    assert "[FAIL-FAST]" in str(exc.value)

    # Test clean close_all
    mgr.close_all()
    assert len(mgr._sessions) == 0


def test_multilingual_engine_clean_shutdown():
    engine = MultilingualNeuralEngine(sample_rate=22050)
    mock_ko = MockSubEngine("ko")
    mock_ko.close = MagicMock()
    engine.register_engine("ko", mock_ko)

    # Context manager exit calls close()
    with engine as active_engine:
        assert not active_engine._is_closed

    assert engine._is_closed
    mock_ko.close.assert_called_once()

