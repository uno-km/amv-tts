# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Expressive Presets & Non-Verbal Acoustic Tokens Audit Suite for termux-tts.
Tests Fast, Balanced, Expressive, Ultra presets and [laugh], [sigh], [breath], [clears_throat] tags.
"""

import time
import pytest
import numpy as np
import termux_tts as tts
from termux_tts.engine import QUALITY_PRESETS, load

from termux_tts.exceptions import TTSModelLoadError, TTSInferenceError

def test_all_four_presets_synthesis():
    try:
        load(language="ko")
    except (TTSModelLoadError, TTSInferenceError):
        pytest.skip("Neural models not installed in local host environment")
    for preset_name in ["fast", "balanced", "expressive", "ultra"]:
        t0 = time.perf_counter()
        with load(language="ko", preset=preset_name) as engine:
            res = engine.synthesize(f"현재 {preset_name} 품질 프리셋으로 음성을 합성 중입니다.")
            assert res.preset == preset_name
            assert res.sample_rate == QUALITY_PRESETS[preset_name]["sample_rate"]
            assert res.duration_sec > 0.3
            assert len(res.wav_bytes) > 1000
            print(f"\n[PASS PRESET: {preset_name:10s}] Duration={res.duration_sec:.2f}s | Latency={res.elapsed_ms:.1f}ms | Rate={res.sample_rate}Hz")

def test_expressive_breath_and_laugh_tokens():
    try:
        load(language="ko", preset="expressive")
    except (TTSModelLoadError, TTSInferenceError):
        pytest.skip("Neural models not installed in local host environment")
    expressive_text = "[clears_throat] 으흠! 안녕하세요 [laugh] 하하하! 오늘 날씨가 참 좋습니다 [sigh] 휴... [breath]"
    with load(language="ko", preset="expressive") as engine:
        res = engine.synthesize(expressive_text, output="expressive_demo.wav")
        assert res.duration_sec > 1.0
        assert res.sample_rate == 24000
        print(f"\n[PASS EXPRESSIVE SYNTHESIS] Synthesized in {res.elapsed_ms:.1f}ms | Duration={res.duration_sec:.2f}s")

def test_tier4_expressive_engine_tags():
    from termux_tts.engine_expressive import ExpressiveEngine
    # Mock base engine for pure unit testing without binary dependency
    class MockSherpa:
        model_name = "mock-vits"
        sample_rate = 22050
        def synthesize(self, text, speed=1.0):
            from termux_tts.audio import AudioBuffer
            from termux_tts.engine_sherpa import SherpaResult
            dur = max(0.5, len(text) * 0.08)
            samples = np.zeros(int(22050 * dur), dtype=np.float32)
            buf = AudioBuffer(samples, sample_rate=22050)
            return SherpaResult(
                text=text, audio_buffer=buf, sample_rate=22050,
                duration_sec=dur, elapsed_ms=10.0, rtf=0.02,
                model_name="mock-vits", backend="MOCK", device_model="mock"
            )
        def close(self):
            pass

    exp_engine = ExpressiveEngine(base_engine=MockSherpa(), sample_rate=22050)
    res = exp_engine.synthesize("안녕하세요 [sigh] 휴우 [laugh] 하하")
    assert "[sigh]" in res.expressive_tags_detected
    assert "[laugh]" in res.expressive_tags_detected
    assert res.duration_sec > 1.0
    assert res.backend == "EXPRESSIVE_NEURAL_ON_DEVICE"
    assert len(res.wav_bytes) > 2000
    print(f"\n[PASS TIER 4 EXPRESSIVE] Tags={res.expressive_tags_detected} | Duration={res.duration_sec:.2f}s")
