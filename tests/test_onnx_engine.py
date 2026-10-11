# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Unit and Integration Tests for Real ONNX Neural Inference Engine & Gateway Routing.
"""

import os
import pytest
import numpy as np

import termux_tts as tts
from termux_tts.engine_onnx import ONNXNeuralEngine
from termux_tts.exceptions import TTSModelLoadError, TTSInferenceError

def test_onnx_engine_missing_model_fail_fast():
    """Verify ONNXNeuralEngine raises TTSModelLoadError when model_path is None or onnxruntime missing."""
    with pytest.raises(TTSModelLoadError) as excinfo:
        ONNXNeuralEngine(model_path=None)
    err_str = str(excinfo.value)
    assert any(k in err_str for k in ("model_path", "onnxruntime", "binary", "assets"))

def test_onnx_engine_nonexistent_file_fail_fast():
    """Verify ONNXNeuralEngine raises TTSModelLoadError when file does not exist or binary missing."""
    with pytest.raises(TTSModelLoadError) as excinfo:
        ONNXNeuralEngine(model_path="nonexistent_vits_model.onnx")
    err_str = str(excinfo.value)
    assert any(k in err_str for k in ("does not exist", "not found", "onnxruntime", "binary", "assets"))

def test_gateway_auto_mode_fails_fast_when_no_models():
    """Verify TTSEngine in auto mode fails fast if neural models are missing (Zero-Silent-Fallback)."""
    with pytest.raises(TTSModelLoadError) as excinfo:
        with tts.load(engine="auto", language="nonexistent_lang") as engine:
            engine.synthesize("테스트 문장")
    assert "[FAIL-FAST]" in str(excinfo.value)

def test_gateway_explicit_onnx_mode_requires_model():
    """Verify tts.load(engine='onnx') fails fast if no model file is given."""
    with pytest.raises(TTSModelLoadError):
        with tts.load(engine="onnx", language="ko") as engine:
            engine.synthesize("신경망 테스트")

