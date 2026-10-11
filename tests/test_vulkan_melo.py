# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Unit and integration tests for MeloTTS Vulkan acceleration (Plan 1 NCNN Sliced & Plan 2 MNN Vulkan).
Tests adhere strictly to AOSF-ENG-STD-2026 and Zero-Silent-Fallback [AMEVA-TTS-E001].
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

import termux_tts
from termux_tts.exceptions import TTSModelLoadError, VulkanInitializationError, TTSInferenceError
from termux_tts.engine_vulkan import VulkanNeuralEngine, VulkanResult
from termux_tts.tokenizer_melo import MeloTokenizer
from termux_tts.audio import AudioBuffer


def test_melo_tokenizer_build_inputs():
    """Verify MeloTokenizer parses text into exact tensor structures required by MeloTTS."""
    tok = MeloTokenizer()
    text = "Hello, this is a test for MeloTTS on Vulkan acceleration."
    inputs = tok.build_inputs(text, speed=1.2)

    assert "x" in inputs
    assert "x_lengths" in inputs
    assert "tones" in inputs
    assert "sid" in inputs
    assert "noise_scale" in inputs
    assert "length_scale" in inputs
    assert "noise_scale_w" in inputs

    assert inputs["x"].dtype == np.int32
    assert inputs["x"].shape[0] == 1
    assert inputs["x_lengths"][0] == inputs["x"].shape[1]
    assert inputs["tones"].shape == inputs["x"].shape
    assert inputs["sid"][0] == 0
    assert np.isclose(inputs["length_scale"][0], 1.0 / 1.2, atol=1e-4)


def test_melo_vulkan_fail_fast_on_missing_assets(tmp_path):
    """Verify VulkanNeuralEngine fails fast with AMEVA-TTS-E001 when Melo assets are absent."""
    with pytest.raises(TTSModelLoadError) as exc_info:
        VulkanNeuralEngine(model_path=str(tmp_path), model_type="melo")
    assert "AMEVA-TTS-E001" in str(exc_info.value)
    assert "MeloTTS Vulkan" in str(exc_info.value)


def test_melo_vulkan_plan2_mnn_init_and_synth(tmp_path):
    """Verify Plan 2 (MNN Vulkan) engine initialization and synthesis."""
    fake_mnn = tmp_path / "melo.mnn"
    fake_mnn.write_bytes(b"MNN_MODEL_MOCK_DATA")
    fake_bin = tmp_path / "melo-mnn-cli"
    fake_bin.write_bytes(b"BIN")

    def mock_run(cmd, *args, **kwargs):
        # cmd: [binary, model, params_file, out_bin]
        out_path = cmd[3]
        mock_pcm = np.zeros(44100, dtype=np.float32)
        with open(out_path, "wb") as f:
            f.write(mock_pcm.tobytes())
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = "OK"
        proc.stderr = ""
        return proc

    with patch.object(VulkanNeuralEngine, "_find_binary", return_value=str(fake_bin)):
        with patch("subprocess.run", side_effect=mock_run):
            engine = VulkanNeuralEngine(model_path=str(tmp_path), model_type="melo_mnn")
            assert engine.strategy == "melo_mnn"
            assert engine.backend == "VULKAN_GPU_MNN_MELO"
            assert engine.sample_rate == 44100

            res = engine.synthesize("MeloTTS Vulkan synthesis test")
            assert isinstance(res, VulkanResult)
            assert res.sample_rate == 44100
            assert res.backend == "VULKAN_GPU_MNN_MELO"
            assert res.duration_sec == 1.0
            assert res.audio_buffer is not None


def test_melo_vulkan_plan1_ncnn_fallback_to_plan2(tmp_path):
    """Verify that if Plan 1 NCNN fails to initialize, it seamlessly falls back to Plan 2 MNN."""
    (tmp_path / "melo_decoder.ncnn.param").write_text("7767517")
    (tmp_path / "melo_decoder.ncnn.bin").write_bytes(b"BIN")
    (tmp_path / "melo_encoder.onnx").write_bytes(b"ONNX")
    (tmp_path / "melo.mnn").write_bytes(b"MNN")
    fake_bin = tmp_path / "melo-mnn-cli"
    fake_bin.write_bytes(b"BIN")

    with patch.object(VulkanNeuralEngine, "_find_binary", return_value=str(fake_bin)):
        engine = VulkanNeuralEngine(model_path=str(tmp_path), model_type="melo")
        assert engine.strategy == "melo_mnn"
        assert engine.backend == "VULKAN_GPU_MNN_MELO"


def test_termux_tts_load_melo_vulkan_dispatch(tmp_path):
    """Verify termux_tts.load(engine='melo', device='vulkan') routes to VulkanNeuralEngine."""
    fake_mnn = tmp_path / "melo.mnn"
    fake_mnn.write_bytes(b"MNN_MODEL")
    fake_bin = tmp_path / "melo-mnn-cli"
    fake_bin.write_bytes(b"BIN")

    mock_report = MagicMock()
    mock_report.overall_success = True
    mock_report.recommended_backend = "vulkan"
    mock_report.device_name = "Mock Vulkan GPU"
    mock_report.vendor_id = 0x10DE

    mock_adapter = MagicMock()
    mock_adapter.return_value.resolve_diagnostic_report.return_value = mock_report

    with patch.object(VulkanNeuralEngine, "_find_binary", return_value=str(fake_bin)):
        with patch("termux_tts.hardware._resolve_ameva_runtime", return_value=MagicMock()), patch("ameva_device.runtime_guard.installed_runtime_version", return_value="2.8.1"):
            with patch("termux_tts.hardware.bind_tts_hardware", return_value=None):
                with patch.dict("sys.modules", {"ameva_runtime.adapters": MagicMock(TtsAdapter=mock_adapter)}):
                    eng = termux_tts.load(model=str(tmp_path), engine="melo", device="vulkan")
                    assert isinstance(eng.synth_engine, VulkanNeuralEngine)
                    assert eng.synth_engine.model_type == "melo"
