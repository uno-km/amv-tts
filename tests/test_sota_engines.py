# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Unit test suite for Tier 5 SOTA Neural Voice Cloning & Expressive Flow Synthesizers:
- CosyVoice 2 (0.5B Audio LLM + DiT Flow Matching)
- F5-TTS (Non-autoregressive Flow Matching)
- ChatTTS (Conversational Auto-regressive)
"""
import os
import pytest
import numpy as np
from pathlib import Path

from termux_tts.engine_sota import SOTANeuralEngine, SOTABackend, SOTAResult
from termux_tts.engine import TTSEngine, load
from termux_tts.exceptions import TTSConfigurationError, TTSInferenceError


class TestSOTAEngines:
    """Rigorous verification of SOTA TTS backends and voice cloning pipelines."""

    def test_backend_enum_resolution(self):
        """Verify alias resolution for all 3 BigTech SOTA architectures."""
        e_cosy = SOTANeuralEngine(backend="cosyvoice")
        assert e_cosy.backend == SOTABackend.COSYVOICE2
        assert e_cosy.sample_rate == 24000

        e_cosy2 = SOTANeuralEngine(backend="cosyvoice2")
        assert e_cosy2.backend == SOTABackend.COSYVOICE2

        e_f5 = SOTANeuralEngine(backend="f5tts")
        assert e_f5.backend == SOTABackend.F5_TTS
        assert e_f5.sample_rate == 24000

        e_chat = SOTANeuralEngine(backend="chattts")
        assert e_chat.backend == SOTABackend.CHATTTS
        assert e_chat.sample_rate == 24000

    def test_invalid_backend_rejection(self):
        """Reject unsupported backend strings with fail-fast exception."""
        with pytest.raises(TTSConfigurationError):
            SOTANeuralEngine(backend="unknown_engine_xyz")

    def test_expressive_tag_normalization(self):
        """Verify conversational non-verbal vocalization tags (breath, sigh, laugh, oral)."""
        engine = SOTANeuralEngine(backend="cosyvoice2")
        sample_text = "물이 계속 솟는 땅에 [breath] 조선이 궁궐을 [침] 세운 방법 [pause] [sigh] 대단하지 않습니까? [laugh]"
        norm_text, tags = engine.normalize_expressive_tags(sample_text)

        assert "[breath]" in tags
        assert "[침]" in tags
        assert "[pause]" in tags
        assert "[sigh]" in tags
        assert "[laugh]" in tags
        assert "<|breath|>" in norm_text
        assert "[oral_1]" in norm_text
        assert "<|pause|>" in norm_text
        assert "<|sigh|>" in norm_text

    def test_cosyvoice2_synthesis_pipeline(self, tmp_path):
        """Verify end-to-end CosyVoice 2 synthesis with ref_audio and tags."""
        engine = SOTANeuralEngine(backend="cosyvoice2", language="ko")
        out_wav = str(tmp_path / "cosy_out.wav")
        fake_ref = str(tmp_path / "ref_voice.wav")
        with open(fake_ref, "wb") as f:
            f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

        res = engine.synthesize(
            text="조선이 한양에 경복궁을 지을 때 [breath] 땅 밑에서 물이 솟구쳤습니다.",
            output=out_wav,
            ref_audio=fake_ref,
            ref_text="조선 왕조의 건국 비화",
            prompt="tense historical mystery narrator tone",
        )

        assert isinstance(res, SOTAResult)
        assert res.backend == "cosyvoice2"
        assert res.sample_rate == 24000
        assert res.duration_sec > 0.5
        assert len(res.audio_buffer.samples) > 0
        assert os.path.exists(out_wav)
        assert "[breath]" in res.expressive_tags_detected

    def test_f5_tts_synthesis_pipeline(self, tmp_path):
        """Verify F5-TTS non-autoregressive flow matching pipeline."""
        engine = SOTANeuralEngine(backend="f5tts", language="en")
        out_wav = str(tmp_path / "f5_out.wav")

        res = engine.synthesize(
            text="How Joseon built the palace on spring ground. [gasp] Unbelievable engineering.",
            output=out_wav,
            speed=1.1,
        )

        assert isinstance(res, SOTAResult)
        assert res.backend == "f5tts"
        assert res.sample_rate == 24000
        assert res.duration_sec > 0.3
        assert os.path.exists(out_wav)

    def test_chattts_conversational_pipeline(self, tmp_path):
        """Verify ChatTTS conversational voice pipeline."""
        engine = SOTANeuralEngine(backend="chattts", language="zh")
        out_wav = str(tmp_path / "chat_out.wav")

        res = engine.synthesize(
            text="朝鲜王朝在建宫殿的时候，地下涌出了泉水。[laugh] 真的很神奇。",
            output=out_wav,
            prompt="lively conversational storyteller",
        )

        assert isinstance(res, SOTAResult)
        assert res.backend == "chattts"
        assert res.sample_rate == 24000
        assert os.path.exists(out_wav)

    def test_tts_engine_unified_gateway_integration(self):
        """Verify top-level TTSEngine and load() factory integration."""
        engine_cosy = load(engine="cosyvoice", preset="cinematic")
        assert engine_cosy.requested_engine_type == "cosyvoice"
        assert isinstance(engine_cosy.synth_engine, SOTANeuralEngine)

        engine_f5 = load(engine="f5tts", preset="sota")
        assert engine_f5.requested_engine_type == "f5tts"
        assert isinstance(engine_f5.synth_engine, SOTANeuralEngine)

        engine_chat = load(engine="chattts")
        assert engine_chat.requested_engine_type == "chattts"
        assert isinstance(engine_chat.synth_engine, SOTANeuralEngine)

        # Synthesize via unified facade
        res = engine_cosy.synthesize(
            "유튜브 숏츠 내레이션 테스트입니다. [breath] 자연스럽게 숨을 쉽니다.",
            speed=1.0,
        )
        assert isinstance(res, SOTAResult)
        assert res.backend == "cosyvoice2"

    def test_empty_text_fail_fast(self):
        """Ensure empty text fails fast without silent mock."""
        engine = SOTANeuralEngine(backend="cosyvoice")
        with pytest.raises(TTSInferenceError):
            engine.synthesize("   ")
