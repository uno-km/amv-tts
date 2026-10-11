# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Android OS System Native TTS Engine Bridge (Option B).
Directly communicates with Samsung / Google Voice Engine via Termux IPC.
Provides authentic human speech output with zero download overhead.
"""

import os
import time
import subprocess
import shutil
from dataclasses import dataclass
from typing import Optional

from .exceptions import TTSInferenceError

@dataclass
class NativeResult:
    text: str
    output_path: Optional[str]
    language: str
    pitch: float
    rate: float
    elapsed_ms: float
    engine_name: str

class NativeAndroidEngine:
    """Android System Native TTS Engine Bridge (Samsung / Google Voice)."""

    def __init__(
        self,
        language: str = "ko",
        pitch: float = 1.0,
        rate: float = 1.0,
        stream: str = "MUSIC"
    ):
        self.language = language
        self.pitch = pitch
        self.rate = rate
        self.stream = stream
        self.binary = self._find_binary()

    def _find_binary(self) -> Optional[str]:
        bin_path = shutil.which("termux-tts-speak")
        if bin_path and os.access(bin_path, os.X_OK):
            return bin_path
        default_p = "/data/data/com.termux/files/usr/bin/termux-tts-speak"
        if os.path.exists(default_p) and os.access(default_p, os.X_OK):
            return default_p
        return None

    def speak(self, text: str, stream: Optional[str] = None) -> NativeResult:
        """Speak text directly through physical Android device speakers via termux-tts-speak IPC."""
        if not text or not text.strip():
            raise TTSInferenceError("Input text cannot be empty or whitespace only.")

        if not self.binary:
            raise TTSInferenceError(
                "[FAIL-FAST] 'termux-tts-speak' binary not found on this system. "
                "NativeAndroidEngine requires an Android Termux environment with 'termux-api' installed (run 'pkg install termux-api'). "
                "If running on non-Android Linux/macOS/Windows, please use '--engine dsp' or '--engine onnx'."
            )

        t0 = time.perf_counter()
        target_stream = stream or self.stream
        cmd = [
            self.binary,
            "-l", self.language,
            "-p", str(self.pitch),
            "-r", str(self.rate),
            "-s", target_stream,
            text
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if res.returncode != 0:
                err_msg = res.stderr.strip() or f"Process exited with code {res.returncode}"
                raise TTSInferenceError(f"Native Android TTS engine execution failed: {err_msg}")
        except subprocess.TimeoutExpired as e:
            raise TTSInferenceError(f"Native Android TTS execution timed out (30s): {e}") from e
        except Exception as e:
            raise TTSInferenceError(f"Failed to execute native Android TTS: {e}") from e

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return NativeResult(
            text=text,
            output_path=None,
            language=self.language,
            pitch=self.pitch,
            rate=self.rate,
            elapsed_ms=elapsed_ms,
            engine_name="Android_Native_Voice_Engine"
        )

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


