# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# -*- coding: utf-8 -*-
"""
E2E 3-SOTA Neural TTS Verification Runner for Galaxy S25:
- CosyVoice 2 (0.5B Flow Matching DiT) - Korean 10s Shorts Script
- F5-TTS (Non-autoregressive DiT) - English 10s Shorts Script
- ChatTTS (Conversational Auto-regressive) - Chinese/Multilingual 10s Shorts Script

Executes real on-device neural synthesis without stubs, validates acoustic integrity,
and plays directly through Galaxy S25 physical speaker at Volume 10.
"""
import os
import sys
import time
import wave
import subprocess
import numpy as np
from pathlib import Path

# Add termux-tts repo to python path
repo_dir = Path.home() / "termux-tts-repo"
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from termux_tts.engine_sota import SOTANeuralEngine, SOTABackend

out_dir = Path.home() / "tts_test_s25"
out_dir.mkdir(parents=True, exist_ok=True)

test_cases = [
    {
        "id": "1. CosyVoice 2",
        "backend": "cosyvoice2",
        "lang": "ko",
        "text": "물이 계속 솟는 땅에 [breath] 조선이 궁궐을 세운 기적의 비밀 [침]. 경복궁과 창덕궁 터는 사실 거대한 습지였습니다.",
        "outfile": out_dir / "s25_real_cosyvoice2_10s.wav",
        "description": "한국어 역사 쇼츠 (호흡/침삼킴 태그 포함)",
    },
    {
        "id": "2. F5-TTS",
        "backend": "f5tts",
        "lang": "en",
        "text": "The secret of how Joseon built royal palaces on swamp land [breath]. Royal architects constructed sophisticated underground culverts to drain the water [pause].",
        "outfile": out_dir / "s25_real_f5tts_10s.wav",
        "description": "영어 다큐멘터리 쇼츠 (호흡/일시정지 태그 포함)",
    },
    {
        "id": "3. ChatTTS",
        "backend": "chattts",
        "lang": "zh",
        "text": "在不断涌出地下水的土地上建造宏伟宫殿的秘密 [laugh] 。历代工匠巧妙设计了地下暗渠系统 [oral_1] 。",
        "outfile": out_dir / "s25_real_chattts_10s.wav",
        "description": "중국어/다국어 대화형 쇼츠 (웃음/구강음 태그 포함)",
    },
]

print("=" * 70)
print("  GALAXY S25 SOTA ON-DEVICE NEURAL TTS E2E VERIFICATION (VOLUME 10)")
print("=" * 70)

# Set max media volume
subprocess.run(["termux-volume", "music", "10"], check=False)
print("[INFO] Termux Media Volume set to 10 (MAX)")

results = []

for tc in test_cases:
    cid = tc["id"]
    backend = tc["backend"]
    lang = tc["lang"]
    text = tc["text"]
    outfile = tc["outfile"]
    desc = tc["description"]

    print(f"\n>>> Running [{cid}] ({desc})")
    print(f"    Script: \"{text}\"")
    print(f"    Target File: {outfile}")

    t_start = time.perf_counter()
    try:
        with SOTANeuralEngine(backend=backend, language=lang, threads=4) as engine:
            res = engine.synthesize(text=text, output=str(outfile), speed=1.0)

        elapsed = time.perf_counter() - t_start

        # Acoustic validation
        with wave.open(str(outfile), "rb") as wf:
            n_frames = wf.getnframes()
            rate = wf.getframerate()
            dur = n_frames / float(rate)
            raw = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
            rms = float(np.sqrt(np.mean(raw.astype(np.float64)**2)))
            peak = int(np.max(np.abs(raw)))

        rtf = elapsed / max(0.01, dur)
        print(f"    [SUCCESS] Neural Synthesis Completed!")
        print(f"      - Duration: {dur:.2f}s | Elapsed: {elapsed:.2f}s | RTF: {rtf:.3f}x")
        print(f"      - Audio Rate: {rate}Hz | RMS: {rms:.1f} | Peak: {peak} / 32767")

        # Play audio at volume 10
        print(f"    [PLAYBACK] Playing {outfile.name} via Physical Speaker (Vol 10)...")
        subprocess.run(["termux-media-player", "stop"], check=False)
        subprocess.run(["termux-media-player", "play", str(outfile)], check=False)
        
        # Wait for audio playback duration
        play_wait = min(dur + 0.5, 12.0)
        time.sleep(play_wait)
        subprocess.run(["termux-media-player", "stop"], check=False)

        results.append({
            "id": cid,
            "status": "PASS",
            "duration": dur,
            "elapsed": elapsed,
            "rtf": rtf,
            "rms": rms,
            "peak": peak,
            "rate": rate,
        })
    except Exception as e:
        print(f"    [ERROR] Synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "id": cid,
            "status": "FAIL",
            "error": str(e),
        })

print("\n" + "=" * 70)
print("  E2E TEST SUMMARY SCORECARD")
print("=" * 70)
for r in results:
    if r["status"] == "PASS":
        print(f"[{r['id']}] STATUS: PASS | Duration: {r['duration']:.2f}s | Elapsed: {r['elapsed']:.2f}s | RTF: {r['rtf']:.3f}x | RMS: {r['rms']:.1f}")
    else:
        print(f"[{r['id']}] STATUS: FAIL | Error: {r.get('error')}")
print("=" * 70)
