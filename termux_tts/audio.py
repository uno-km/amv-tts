# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Zero-Dependency Audio Buffer and RIFF WAV Stream Encoder.
Handles 16-bit Linear PCM formatting with soft-clipping protection.
"""

import io
import wave
from pathlib import Path
from typing import Union
import numpy as np
from .exceptions import TTSAudioEncodingError

class AudioBuffer:
    """Manages raw floating-point audio samples and converts to 16-bit PCM WAV."""

    def __init__(self, samples: Union[np.ndarray, list], sample_rate: int = 22050):
        if isinstance(samples, list):
            self.samples = np.array(samples, dtype=np.float32)
        elif isinstance(samples, np.ndarray):
            self.samples = samples.astype(np.float32).flatten()
        else:
            raise TTSAudioEncodingError("Samples must be a list or numpy ndarray.")

        self.sample_rate = sample_rate
        self._normalize_and_clip()

    def _normalize_and_clip(self) -> None:
        """Apply peak normalization and soft clipping to prevent distortion."""
        if len(self.samples) == 0:
            return

        peak = np.max(np.abs(self.samples))
        if peak > 1.0:
            self.samples = self.samples / peak
        elif peak < 0.0001:
            pass # Keep quiet signals as is
        else:
            # Gentle scale to target -1dB
            self.samples = self.samples * 0.95

    @property
    def duration_seconds(self) -> float:
        """Duration of the audio in seconds."""
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / float(self.sample_rate)

    def to_pcm16_bytes(self) -> bytes:
        """Convert float32 [-1.0, 1.0] samples into 16-bit signed integer bytes."""
        clean = np.nan_to_num(self.samples, nan=0.0, posinf=1.0, neginf=-1.0)
        clipped = np.clip(clean, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    def to_wav_bytes(self) -> bytes:
        """Encode raw samples into standard RIFF WAV format."""
        try:
            pcm_bytes = self.to_pcm16_bytes()
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                wav_file.setnchannels(1)      # Mono
                wav_file.setsampwidth(2)      # 16-bit (2 bytes)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_bytes)
            return buf.getvalue()
        except Exception as e:
            raise TTSAudioEncodingError(f"Failed to encode WAV buffer: {e}") from e

    def pad_silence(self, lead_in_ms: int = 200, lead_out_ms: int = 150) -> "AudioBuffer":
        """Prepend and append true zero-amplitude silence to prevent Android DAC ramp-up clipping."""
        lead_in_samples = int(self.sample_rate * (lead_in_ms / 1000.0))
        lead_out_samples = int(self.sample_rate * (lead_out_ms / 1000.0))
        zeros_in = np.zeros(lead_in_samples, dtype=np.float32)
        zeros_out = np.zeros(lead_out_samples, dtype=np.float32)
        padded = np.concatenate([zeros_in, self.samples, zeros_out])
        return AudioBuffer(padded, sample_rate=self.sample_rate)

    @classmethod
    def from_wav_file(cls, filepath: str) -> "AudioBuffer":
        """Load an AudioBuffer directly from a WAV file on disk."""
        try:
            with wave.open(filepath, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                pcm16 = np.frombuffer(frames, dtype=np.int16)
                samples = (pcm16 / 32767.0).astype(np.float32)
                return cls(samples, sample_rate=sr)
        except Exception as e:
            raise TTSAudioEncodingError(f"Failed to read WAV file '{filepath}': {e}") from e

    def save(self, filepath: str) -> str:
        """Save the audio buffer to a WAV file on disk."""
        target = Path(filepath).expanduser().resolve()
        wav_data = self.to_wav_bytes()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as f:
                f.write(wav_data)
            return str(target)
        except Exception as e:
            raise TTSAudioEncodingError(f"Failed to save WAV to '{filepath}': {e}") from e
