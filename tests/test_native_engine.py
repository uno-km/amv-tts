# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Unit Tests for Option B (Android Native System Voice Engine Bridge).
"""

import pytest
import termux_tts as tts
from termux_tts.engine_native import NativeAndroidEngine
from termux_tts.exceptions import TTSInferenceError

def test_native_engine_initialization():
    engine = NativeAndroidEngine(language="ko", pitch=1.0, rate=1.0, stream="MUSIC")
    assert engine.language == "ko"
    assert engine.stream == "MUSIC"

def test_native_engine_speak_empty_guard():
    engine = NativeAndroidEngine(language="ko")
    with pytest.raises(TTSInferenceError):
        engine.speak("")
    with pytest.raises(TTSInferenceError):
        engine.speak("   ")

def test_native_engine_speak_execution():
    with tts.load(engine="native", language="ko") as engine:
        if not engine.binary:
            # 1. Non-Android host must fail-fast with strict TTSInferenceError (Zero-Fallback)
            with pytest.raises(TTSInferenceError) as excinfo:
                engine.speak("테스트 발화입니다.")
            assert "FAIL-FAST" in str(excinfo.value)
            print(f"\n[PASS NATIVE FAIL-FAST] Correctly rejected: {excinfo.value}")
        else:
            # 2. Genuine Android Termux system execution
            res = engine.speak("테스트 발화입니다.")
            assert res.engine_name == "Android_Native_Voice_Engine"
            assert res.language == "ko"
            assert res.elapsed_ms >= 0.0
            print(f"\n[PASS NATIVE SPEAK] Engine: {res.engine_name} | Elapsed: {res.elapsed_ms:.2f}ms")


