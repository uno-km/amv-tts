# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest

from termux_tts.engine import TTSEngine, load
from termux_tts.engine_sherpa import SherpaNeuralEngine
from termux_tts.cli import main


def test_tts_engine_load_rpc_parameters():
    with patch("termux_tts.engine.SherpaNeuralEngine") as MockSherpa:
        mock_sherpa_inst = MagicMock()
        MockSherpa.return_value = mock_sherpa_inst

        with patch("termux_tts.engine.bind_tts_hardware"):
            engine = load(
                engine="neural",
                rpc="192.168.0.220:50052,192.168.0.253:50052",
                tensor_split="50,50",
            )
            assert engine.rpc == "192.168.0.220:50052,192.168.0.253:50052"
            assert engine.tensor_split == "50,50"
            assert MockSherpa.call_count == 1
            _, kwargs = MockSherpa.call_args
            assert kwargs.get("rpc") == "192.168.0.220:50052,192.168.0.253:50052"
            assert kwargs.get("tensor_split") == "50,50"


def test_tts_cli_rpc_arg_parsing(monkeypatch):
    test_args = [
        "termux-tts",
        "synth",
        "-t",
        "Hello RPC synthesis",
        "--rpc",
        "192.168.0.220:50052,192.168.0.253:50052",
        "-ts",
        "60,40",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("termux_tts.cli.load") as mock_load:
        mock_engine = MagicMock()
        mock_res = MagicMock(
            backend="SHERPA_RPC",
            model_name="vits-rpc",
            duration_sec=2.1,
            elapsed_ms=120.0,
            rtf=0.05,
        )
        mock_engine.synthesize.return_value = mock_res
        mock_engine.__enter__.return_value = mock_engine
        mock_engine.__exit__.return_value = None
        mock_load.return_value = mock_engine

        main()

        assert mock_load.call_count == 1
        _, kwargs = mock_load.call_args
        assert kwargs.get("rpc") == "192.168.0.220:50052,192.168.0.253:50052"
        assert kwargs.get("tensor_split") == "60,40"


def test_sherpa_engine_rpc_cmd_injection(tmp_path):
    with patch.object(SherpaNeuralEngine, "_find_binary", return_value="/mock/sherpa-onnx-offline-tts"), \
         patch.object(SherpaNeuralEngine, "_resolve_model_assets", return_value={
             "model": str(tmp_path / "model.onnx"),
             "tokens": str(tmp_path / "tokens.txt"),
             "model_name": "mock-vits",
         }), \
         patch("termux_tts.engine_sherpa.verify_rpc_cluster_nodes"):

        engine = SherpaNeuralEngine(
            model_path=str(tmp_path),
            rpc="192.168.0.220:50052,192.168.0.253:50052",
            tensor_split="40,30,30",
        )

        with patch("subprocess.run") as mock_run, \
             patch("termux_tts.engine_sherpa.AudioBuffer") as mock_ab:
            def fake_run(cmd, *args, **kwargs):
                for arg in cmd:
                    if arg.startswith("--output-filename="):
                        fn = arg.split("=", 1)[1]
                        with open(fn, "wb") as f:
                            f.write(b"RIFFmockwavdata12345678")
                mock_res = MagicMock()
                mock_res.returncode = 0
                mock_res.stderr = ""
                return mock_res

            mock_run.side_effect = fake_run

            # Mock AudioBuffer.from_wav_file
            mock_buf = MagicMock(duration=1.5, duration_seconds=1.5, sample_rate=22050)
            mock_buf.pad_silence.return_value = mock_buf
            mock_ab.from_wav_file.return_value = mock_buf

            with patch("os.unlink"):
                engine.synthesize("Test speech prompt")

            assert mock_run.call_count >= 1
            cmd = mock_run.call_args[0][0]
            assert "--rpc" in cmd
            rpc_idx = cmd.index("--rpc")
            assert cmd[rpc_idx + 1] == "192.168.0.220:50052,192.168.0.253:50052"

            assert "--tensor-split" in cmd
            ts_idx = cmd.index("--tensor-split")
            assert cmd[ts_idx + 1] == "40,30,30"
