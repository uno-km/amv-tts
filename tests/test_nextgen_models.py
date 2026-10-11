# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import termux_tts
from termux_tts.exceptions import TTSModelLoadError, TTSInferenceError
from termux_tts.engine_sherpa import SherpaNeuralEngine


def test_kokoro_model_search_fail_fast(tmp_path):
    """Test Kokoro fails fast with helpful instruction when assets are missing."""
    with patch.object(SherpaNeuralEngine, "_find_binary", return_value="/mock/bin"):
        with pytest.raises(TTSModelLoadError) as exc_info:
            SherpaNeuralEngine(model_path=str(tmp_path), model_type="kokoro")
        assert "Kokoro model assets" in str(exc_info.value)
        assert "termux-tts install --models kokoro" in str(exc_info.value)


def test_supertonic_model_search_fail_fast(tmp_path):
    """Test Supertonic fails fast when assets are missing."""
    with patch.object(SherpaNeuralEngine, "_find_binary", return_value="/mock/bin"):
        with pytest.raises(TTSModelLoadError) as exc_info:
            SherpaNeuralEngine(model_path=str(tmp_path), model_type="supertonic")
        assert "Supertonic model assets" in str(exc_info.value)
        assert "termux-tts install --models supertonic" in str(exc_info.value)


def test_melo_model_search_fail_fast(tmp_path):
    """Test MeloTTS fails fast when assets are missing."""
    with patch.object(SherpaNeuralEngine, "_find_binary", return_value="/mock/bin"):
        with pytest.raises(TTSModelLoadError) as exc_info:
            SherpaNeuralEngine(model_path=str(tmp_path), model_type="melo")
        assert "MeloTTS model assets" in str(exc_info.value)
        assert "termux-tts install --models melo" in str(exc_info.value)


def test_kokoro_model_asset_resolution(tmp_path):
    """Test Kokoro resolves assets correctly when all files exist."""
    k_dir = tmp_path / "kokoro-int8-en-v0_19"
    k_dir.mkdir()
    (k_dir / "model.int8.onnx").write_text("fake onnx")
    (k_dir / "voices.bin").write_text("fake voices")
    (k_dir / "tokens.txt").write_text("fake tokens")
    espeak_dir = k_dir / "espeak-ng-data"
    espeak_dir.mkdir()

    with patch.object(SherpaNeuralEngine, "_find_binary", return_value="/mock/bin"):
        eng = SherpaNeuralEngine(model_path=str(k_dir), model_type="kokoro")
        assert eng.model_assets["model"] == str(k_dir / "model.int8.onnx")
        assert eng.model_assets["voices"] == str(k_dir / "voices.bin")
        assert eng.model_assets["tokens"] == str(k_dir / "tokens.txt")
        assert eng.model_assets["data_dir"] == str(espeak_dir)


def test_supertonic_model_asset_resolution(tmp_path):
    """Test Supertonic resolves all 7 required files correctly."""
    s_dir = tmp_path / "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
    s_dir.mkdir()
    (s_dir / "duration_predictor.int8.onnx").write_text("fake dp")
    (s_dir / "text_encoder.int8.onnx").write_text("fake te")
    (s_dir / "vector_estimator.int8.onnx").write_text("fake ve")
    (s_dir / "vocoder.int8.onnx").write_text("fake voc")
    (s_dir / "tts.json").write_text("{}")
    (s_dir / "unicode_indexer.bin").write_text("fake ui")
    (s_dir / "voice.bin").write_text("fake vs")

    with patch.object(SherpaNeuralEngine, "_find_binary", return_value="/mock/bin"):
        eng = SherpaNeuralEngine(model_path=str(s_dir), model_type="supertonic")
        assert eng.model_assets["duration_predictor"] == str(s_dir / "duration_predictor.int8.onnx")
        assert eng.model_assets["text_encoder"] == str(s_dir / "text_encoder.int8.onnx")
        assert eng.model_assets["voice_style"] == str(s_dir / "voice.bin")


def test_melo_model_asset_resolution(tmp_path):
    """Test MeloTTS resolves assets correctly."""
    m_dir = tmp_path / "vits-melo-tts-zh_en"
    m_dir.mkdir()
    (m_dir / "model.onnx").write_text("fake onnx")
    (m_dir / "tokens.txt").write_text("fake tokens")
    (m_dir / "lexicon.txt").write_text("fake lexicon")

    with patch.object(SherpaNeuralEngine, "_find_binary", return_value="/mock/bin"):
        eng = SherpaNeuralEngine(model_path=str(m_dir), model_type="melo")
        assert eng.model_assets["model"] == str(m_dir / "model.onnx")
        assert eng.model_assets["tokens"] == str(m_dir / "tokens.txt")
        assert eng.model_assets["lexicon"] == str(m_dir / "lexicon.txt")


def test_gateway_load_dispatch_nextgen(tmp_path):
    """Test unified gateway termux_tts.load() dispatches to nextgen engines."""
    k_dir = tmp_path / "kokoro-int8-en-v0_19"
    k_dir.mkdir()
    (k_dir / "model.onnx").write_text("fake onnx")
    (k_dir / "voices.bin").write_text("fake voices")
    (k_dir / "tokens.txt").write_text("fake tokens")

    with patch.object(SherpaNeuralEngine, "_find_binary", return_value="/mock/bin"):
        eng = termux_tts.load(engine="kokoro", model=str(k_dir))
        assert isinstance(eng.synth_engine, SherpaNeuralEngine)
        assert eng.synth_engine.model_type == "kokoro"
