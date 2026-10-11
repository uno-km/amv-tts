# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
AMEVA Unified Distributed RPC Cluster Orchestration Cold Tests for termux-tts.
Strict Protocol: Zero-Silent-Fallback & Fail-Fast Verification.
Component: [TTS-CLUSTER]
"""

import socket
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

import termux_tts as tts
from termux_tts.exceptions import (
    ClusterConnectionError,
    ClusterConfigurationError,
)
from termux_tts.cluster import (
    parse_cluster_rpc_spec,
    verify_rpc_cluster_nodes,
    verify_rpc_cluster_health,
)
from termux_tts.engine import TTSEngine, load
from termux_tts.engine_sherpa import SherpaNeuralEngine
from termux_tts.cli import main


def test_parse_cluster_rpc_spec_valid():
    """Verify parsing and normalization of valid RPC specs."""
    servers_str = "192.0.2.11:50052, 192.0.2.12:50052"
    parsed = parse_cluster_rpc_spec(servers_str)
    assert parsed == ["192.0.2.11:50052", "192.0.2.12:50052"]

    servers_list = ["192.0.2.11:50052", "192.0.2.12:50052"]
    parsed2 = parse_cluster_rpc_spec(servers_list)
    assert parsed2 == ["192.0.2.11:50052", "192.0.2.12:50052"]

    assert parse_cluster_rpc_spec(None) == []
    assert parse_cluster_rpc_spec("") == []


def test_parse_cluster_rpc_spec_invalid():
    """Verify invalid RPC formats strictly raise ClusterConfigurationError."""
    with pytest.raises(ClusterConfigurationError):
        parse_cluster_rpc_spec("invalid_host_no_port")

    with pytest.raises(ClusterConfigurationError):
        parse_cluster_rpc_spec("host:not_a_number")

    with pytest.raises(ClusterConfigurationError):
        parse_cluster_rpc_spec("host:999999")  # Port out of range

    with pytest.raises(ClusterConfigurationError):
        parse_cluster_rpc_spec(12345)  # Invalid type


def test_verify_rpc_cluster_nodes_success():
    """Verify pre-flight check succeeds when all RPC nodes accept TCP connection."""
    servers = ["192.0.2.11:50052", "192.0.2.12:50052"]
    with patch("socket.create_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value = MagicMock()
        verify_rpc_cluster_nodes(servers, timeout=1.0)
        assert mock_conn.call_count == 2


def test_verify_rpc_cluster_nodes_fail_fast():
    """Verify Zero-Silent-Fallback: Unreachable node raises ClusterConnectionError immediately."""
    servers = ["192.0.2.11:50052", "192.0.2.12:50052"]

    def fake_connect(addr, timeout=None):
        host, port = addr
        if host == "192.0.2.12":
            raise ConnectionRefusedError("Connection refused by test worker")
        return MagicMock()

    with patch("socket.create_connection", side_effect=fake_connect):
        with pytest.raises(ClusterConnectionError) as exc_info:
            verify_rpc_cluster_nodes(servers, timeout=1.0)
        assert "Zero-Silent-Fallback Violation Prevented" in str(exc_info.value)
        assert "192.0.2.12:50052" in str(exc_info.value)


def test_verify_rpc_cluster_health_reporting():
    """Verify granular diagnostics from verify_rpc_cluster_health."""
    servers = ["192.0.2.11:50052", "192.0.2.12:50052"]

    def fake_connect(addr, timeout=None):
        host, port = addr
        if host == "192.0.2.12":
            raise socket.timeout("Timed out")
        return MagicMock()

    with patch("socket.create_connection", side_effect=fake_connect):
        health = verify_rpc_cluster_health(servers, timeout=1.0)
        assert health["all_healthy"] is False
        assert health["nodes"]["192.0.2.11:50052"]["reachable"] is True
        assert health["nodes"]["192.0.2.12:50052"]["reachable"] is False


def test_tts_engine_load_cluster_kwargs():
    """Verify load() correctly passes cluster kwargs to SherpaNeuralEngine."""
    with patch("termux_tts.engine.SherpaNeuralEngine") as MockSherpa, \
         patch("termux_tts.engine.bind_tts_hardware"):
        mock_sherpa_inst = MagicMock()
        MockSherpa.return_value = mock_sherpa_inst

        engine = load(
            engine="neural",
            cluster_rpc_servers="192.0.2.11:50052,192.0.2.12:50052",
            cluster_tensor_split="50,50",
            cluster_split_mode="tensor",
            cluster_vram_budget="4096,4096",
        )
        assert MockSherpa.call_count == 1
        _, kwargs = MockSherpa.call_args
        assert kwargs.get("cluster_rpc_servers") == ["192.0.2.11:50052", "192.0.2.12:50052"]
        assert kwargs.get("cluster_tensor_split") == "50,50"
        assert kwargs.get("cluster_split_mode") == "tensor"
        assert kwargs.get("cluster_vram_budget") == "4096,4096"


def test_sherpa_neural_engine_cluster_args_and_preflight(tmp_path):
    """Verify SherpaNeuralEngine preflight check and command injection."""
    with patch.object(SherpaNeuralEngine, "_find_binary", return_value="/mock/sherpa-onnx-offline-tts"), \
         patch.object(SherpaNeuralEngine, "_resolve_model_assets", return_value={
             "model": str(tmp_path / "model.onnx"),
             "tokens": str(tmp_path / "tokens.txt"),
             "model_name": "mock-vits",
         }), \
         patch("termux_tts.engine_sherpa.verify_rpc_cluster_nodes") as mock_verify:

        engine = SherpaNeuralEngine(
            model_path=str(tmp_path),
            cluster_rpc_servers="192.0.2.11:50052,192.0.2.12:50052",
            cluster_tensor_split="50,50",
        )
        assert engine.cluster_rpc_servers == ["192.0.2.11:50052", "192.0.2.12:50052"]
        assert engine.rpc == "192.0.2.11:50052,192.0.2.12:50052"

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
            mock_buf = MagicMock(duration=1.0, duration_seconds=1.0, sample_rate=22050)
            mock_buf.pad_silence.return_value = mock_buf
            mock_ab.from_wav_file.return_value = mock_buf

            with patch("os.unlink"):
                engine.synthesize("Test cluster prompt")

            assert mock_verify.called
            assert mock_run.call_count >= 1
            cmd = mock_run.call_args[0][0]
            assert "--rpc" in cmd
            rpc_idx = cmd.index("--rpc")
            assert cmd[rpc_idx + 1] == "192.0.2.11:50052,192.0.2.12:50052"
            assert "--tensor-split" in cmd
            ts_idx = cmd.index("--tensor-split")
            assert cmd[ts_idx + 1] == "50,50"


def test_cli_cluster_flags_parsing(monkeypatch):
    """Verify CLI parses cluster arguments."""
    test_args = [
        "termux-tts",
        "synth",
        "-t",
        "Hello Unified Cluster TTS",
        "--cluster-rpc-servers",
        "192.0.2.11:50052,192.0.2.12:50052",
        "--cluster-tensor-split",
        "40,60",
        "--cluster-split-mode",
        "tensor",
        "--cluster-vram-budget",
        "4000,4000",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("termux_tts.cli.load") as mock_load:
        mock_engine = MagicMock()
        mock_res = MagicMock(
            backend="SHERPA_RPC",
            model_name="vits-rpc",
            duration_sec=1.5,
            elapsed_ms=80.0,
            rtf=0.05,
        )
        mock_engine.synthesize.return_value = mock_res
        mock_engine.__enter__.return_value = mock_engine
        mock_engine.__exit__.return_value = None
        mock_load.return_value = mock_engine

        main()

        assert mock_load.call_count == 1
        _, kwargs = mock_load.call_args
        assert kwargs.get("cluster_rpc_servers") == "192.0.2.11:50052,192.0.2.12:50052"
        assert kwargs.get("cluster_tensor_split") == "40,60"
        assert kwargs.get("cluster_split_mode") == "tensor"
        assert kwargs.get("cluster_vram_budget") == "4000,4000"
