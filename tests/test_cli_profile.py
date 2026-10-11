# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Unit tests for termux-tts CLI profile subcommands and --profile flag (SCRUM-529).
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from termux_tts.cli import main
from termux_tts.speaker_profile import SpeakerProfile


class TestCLIProfileRouting(unittest.TestCase):
    def setUp(self):
        self.mock_res = MagicMock()
        self.mock_res.backend = "SHERPA_NEURAL"
        self.mock_res.model_name = "kokoro-82m"
        self.mock_res.duration_sec = 3.2
        self.mock_res.elapsed_ms = 45.0
        self.mock_res.rtf = 0.014

    @patch("termux_tts.cli.load")
    def test_synth_with_profile_flag(self, mock_load):
        """Verify that --profile flag is forwarded to engine.synthesize."""
        mock_engine = MagicMock()
        mock_engine.synthesize.return_value = self.mock_res
        mock_load.return_value.__enter__.return_value = mock_engine

        test_args = ["termux-tts", "synth", "-t", "Voice test", "--profile", "custom_spk"]
        with patch.object(sys, "argv", test_args):
            main()

        mock_engine.synthesize.assert_called_with(
            "Voice test",
            output=unittest.mock.ANY,
            speed=1.0,
            language="auto",
            mode="unified",
            profile="custom_spk",
        )

    @patch("termux_tts.speaker_profile.SpeakerProfileEngine.list_profiles")
    def test_profile_list_subcommand(self, mock_list):
        """Verify profile list execution without error."""
        mock_list.return_value = []
        test_args = ["termux-tts", "profile", "list"]
        with patch.object(sys, "argv", test_args):
            main()
        mock_list.assert_called_once()

    @patch("termux_tts.speaker_profile.SpeakerProfileEngine.get_profile")
    def test_profile_inspect_subcommand(self, mock_get):
        """Verify profile inspect execution."""
        mock_prof = MagicMock()
        mock_prof.name = "alice"
        mock_prof.embedding = [0.1] * 192
        mock_prof.duration_sec = 4.0
        mock_prof.pitch_mean_hz = 220.0
        mock_prof.energy_rms_db = -12.0
        mock_prof.created_at = "2026-10-11T00:00:00Z"
        mock_prof.extractor_type = "campplus_int8"
        mock_prof.source_file = "alice.wav"
        mock_get.return_value = mock_prof

        test_args = ["termux-tts", "profile", "inspect", "alice"]
        with patch.object(sys, "argv", test_args):
            main()
        mock_get.assert_called_with("alice")


if __name__ == "__main__":
    unittest.main()
