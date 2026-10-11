# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Tests for termux-tts CLI zero-config argument routing and ergonomic entrypoints.
"""
import sys
import unittest
from unittest.mock import patch, MagicMock


class TestCLIRouting(unittest.TestCase):
    def setUp(self):
        self.mock_res = MagicMock()
        self.mock_res.backend = "MULTILINGUAL_NEURAL_ORCHESTRATOR"
        self.mock_res.model_name = "hybrid-multilingual-mesh"
        self.mock_res.duration_sec = 2.5
        self.mock_res.elapsed_ms = 120.0
        self.mock_res.rtf = 0.048

    @patch("termux_tts.cli.load")
    def test_positional_text_without_synth(self, mock_load):
        mock_engine = MagicMock()
        mock_engine.synthesize.return_value = self.mock_res
        mock_load.return_value.__enter__.return_value = mock_engine

        test_args = ["termux-tts", "Hello", "world", "positional", "-o", "out.wav"]
        with patch.object(sys, "argv", test_args):
            from termux_tts.cli import main
            main()

        mock_engine.synthesize.assert_called_with(
            "Hello world positional",
            output=unittest.mock.ANY,
            speed=1.0,
            language="auto",
            mode="unified",
        )

    @patch("termux_tts.cli.load")
    def test_flag_text_without_synth(self, mock_load):
        mock_engine = MagicMock()
        mock_engine.synthesize.return_value = self.mock_res
        mock_load.return_value.__enter__.return_value = mock_engine

        test_args = ["termux-tts", "-t", "Explicit flag text", "-o", "flag_out.wav"]
        with patch.object(sys, "argv", test_args):
            from termux_tts.cli import main
            main()

        mock_engine.synthesize.assert_called_with(
            "Explicit flag text",
            output=unittest.mock.ANY,
            speed=1.0,
            language="auto",
            mode="unified",
        )

    @patch("termux_tts.cli.load")
    def test_explicit_synth_subcommand(self, mock_load):
        mock_engine = MagicMock()
        mock_engine.synthesize.return_value = self.mock_res
        mock_load.return_value.__enter__.return_value = mock_engine

        test_args = ["termux-tts", "synth", "-t", "Standard synth", "-l", "ko"]
        with patch.object(sys, "argv", test_args):
            from termux_tts.cli import main
            main()

        mock_engine.synthesize.assert_called_with(
            "Standard synth",
            output=unittest.mock.ANY,
            speed=1.0,
            language="ko",
            mode="unified",
        )

    @patch("termux_tts.cli.load")
    def test_speak_subcommand_routes_to_native(self, mock_load):
        mock_engine = MagicMock()
        mock_engine.speak.return_value = MagicMock(engine_name="NATIVE_SYSTEM", elapsed_ms=15.0)
        mock_load.return_value.__enter__.return_value = mock_engine

        test_args = ["termux-tts", "speak", "-t", "Native speech broadcast", "-s", "MUSIC"]
        with patch.object(sys, "argv", test_args):
            from termux_tts.cli import main
            main()

        mock_engine.speak.assert_called_with("Native speech broadcast", stream="MUSIC")

    @patch("termux_tts.installer.run_installation")
    def test_install_subcommand_routes_cleanly(self, mock_install):
        test_args = ["termux-tts", "install", "--models", "default", "--no-play"]
        with patch.object(sys, "argv", test_args):
            from termux_tts.cli import main
            main()

        mock_install.assert_called_with(
            models="default",
            force=False,
            play=False
        )


if __name__ == "__main__":
    unittest.main()
