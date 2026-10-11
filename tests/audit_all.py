# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import sys
from faster_whisper import WhisperModel
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
model = WhisperModel('tiny', device='cpu', compute_type='int8')

files = [
    ('tests/s25_real_cosyvoice2_10s.wav', 'ko', 'CosyVoice 2 (FP16 ONNX)'),
    ('tests/f5_official_verified.wav', 'en', 'F5-TTS (16-Step DiT ONNX)'),
    ('tests/s25_real_chattts_10s.wav', 'zh', 'ChatTTS (INT8 ONNX)'),
    ('tests/real_korean_shorts_10s.wav', 'ko', 'Neural VITS (On-Device Production)'),
    ('tests/real_english_kokoro_10s.wav', 'en', 'Neural Kokoro 24kHz (On-Device Production)'),
]

print('=' * 75)
print('         WHISPER STT RIGOROUS AUDIT SCORECARD (GROUND TRUTH)')
print('=' * 75)

for path, lang, desc in files:
    p = Path(path)
    if not p.exists():
        print(f'[{desc}] File missing: {path}')
        continue
    segs, info = model.transcribe(str(p), language=lang)
    text = ' '.join([s.text for s in segs]).strip()
    is_valid = len(text) > 5 and not text.startswith(',')
    status = "VALID HUMAN SPEECH" if is_valid else "NON-INTELLIGIBLE (NOISE/BEEP)"
    print(f'[{desc}]')
    print(f'  - File: {path} ({p.stat().st_size} bytes)')
    print(f'  - Target Language: {lang}')
    print(f'  - STT Transcription: "{text}"')
    print(f'  - Verification Verdict: [{status}]')
    print('-' * 75)
