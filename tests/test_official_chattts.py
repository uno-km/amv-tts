# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import ChatTTS
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
import time
import sys

sys.stdout.reconfigure(encoding="utf-8")

print("=" * 65)
print("  OFFICIAL ChatTTS INFERENCE & WHISPER STT AUDIT")
print("=" * 65)

# 1. Initialize ChatTTS
chat = ChatTTS.Chat()
print("[1/4] Loading ChatTTS official model checkpoints from HuggingFace...")
t0 = time.time()
chat.load(source="huggingface", compile=False)
print(f"      Models loaded in {time.time()-t0:.2f}s")

# 2. Text Script with expressive conversational tokens
texts = [
    "在不断涌出地下水的土地上建造宏伟宫殿的秘密 [laugh] 。历代工匠巧妙设计了地下暗渠系统 [oral_1] 。",
    "물이 계속 솟는 땅에 조선이 궁궐을 세운 기적의 비밀 [laugh] 。 경복궁 터는 사실 거대한 습지였습니다 [oral_1] 。"
]

out_files = [
    Path("tests/official_chattts_zh.wav"),
    Path("tests/official_chattts_ko.wav"),
]

langs = ["zh", "ko"]

for i, text in enumerate(texts):
    print(f"\n[2/4] Generating speech for script {i+1}...")
    print(f"      Text: \"{text}\"")
    t_inf = time.time()
    wavs = chat.infer(text, use_decoder=True)
    elapsed = time.time() - t_inf

    audio_data = np.array(wavs[0], dtype=np.float32).squeeze()
    sf.write(str(out_files[i]), audio_data, 24000)
    dur = len(audio_data) / 24000.0
    print(f"      Synthesized in {elapsed:.2f}s (Duration: {dur:.2f}s, RTF: {elapsed/dur:.2f}x)")
    print(f"      Saved: {out_files[i]}")

# 3. Whisper STT Audit
print("\n[3/4] Whisper STT Self-Audit (No External Playback)...")
from faster_whisper import WhisperModel
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")

for i, out_file in enumerate(out_files):
    segs, info = stt_model.transcribe(str(out_file), language=langs[i])
    stt_text = " ".join([s.text for s in segs]).strip()
    print(f"\n[AUDIT {i+1}] File: {out_file.name}")
    print(f"  Detected Language: {info.language} (confidence: {info.language_probability:.2f})")
    print(f"  STT Transcribed Text: \"{stt_text}\"")
    is_valid = len(stt_text) > 3 and not stt_text.startswith(",")
    verdict = "PASS (HUMAN SPEECH)" if is_valid else "FAIL (NOISE)"
    print(f"  Audit Verdict: [{verdict}]")

print("=" * 65)
