# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
from faster_whisper import WhisperModel
import sys
from pathlib import Path

print("Loading Whisper STT model...")
model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("Whisper model loaded.")

def transcribe(audio_path, lang=None):
    if not Path(audio_path).exists():
        print(f"File not found: {audio_path}")
        return ""
    segments, info = model.transcribe(str(audio_path), language=lang, beam_size=3)
    text = " ".join([seg.text for seg in segments]).strip()
    print(f"\n[STT AUDIT] File: {audio_path}")
    print(f"  Detected Language: {info.language} (confidence: {info.language_probability:.2f})")
    print(f"  Transcribed Text: \"{text}\"")
    return text

if __name__ == "__main__":
    transcribe("tests/s25_real_chattts_10s.wav", lang="zh")
    transcribe("tests/final_f5_perfect_10s.wav", lang="en")
    transcribe("tests/s25_real_cosyvoice2_10s.wav", lang="ko")
