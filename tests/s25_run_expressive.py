# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import os
import sys
import time
import re
import wave
import numpy as np
from pathlib import Path

# Add termux-tts repo path
repo_path = Path.home() / "termux-tts-repo"
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

from termux_tts.engine_sherpa import SherpaNeuralEngine
from termux_tts.audio import AudioBuffer

print("=" * 65)
print("  GALAXY S25 EXPRESSIVE NEURAL KOREAN SHORTS SYNTHESIZER")
print("=" * 65)

# Initialize On-Device Neural Engine (Korean KSS)
print("[1/4] Initializing On-Device Korean Neural Acoustic Engine...")
base_engine = SherpaNeuralEngine(
    language="ko",
    device="auto",
    threads=4,
    sample_rate=22050,
    model_type="vits",
)
sr = base_engine.sample_rate
print(f"      Engine ready: model={base_engine.model_name}, sr={sr}Hz")

def generate_inhale_breath(duration_sec=0.35):
    """Organic human inhalation breath (스읍- 들숨)."""
    n = int(sr * duration_sec)
    t = np.linspace(0, 1, n, dtype=np.float32)
    # Organic swell envelope: slow rise, sharp cutoff at the moment of vocal onset
    env = (t ** 1.8) * np.exp(-1.2 * (1.0 - t))
    # Filtered friction noise (focusing on 2000-4500Hz respiratory tract turbulence)
    white = np.random.normal(0, 1.0, n).astype(np.float32)
    # Bandpass approximation via differential smoothing
    turb = white - np.convolve(white, np.ones(5)/5.0, mode='same')
    breath = turb * env * 0.18
    return breath.astype(np.float32)

def generate_swallow_gulp(duration_sec=0.25):
    """Subtle human pharyngeal swallow click with pre/post pause (침 삼킴)."""
    n = int(sr * duration_sec)
    wave_arr = np.zeros(n, dtype=np.float32)
    # Click occurs around 40% into the pause
    click_pos = int(n * 0.35)
    click_len = int(sr * 0.06) # 60ms click & resonance
    t_c = np.linspace(0, 1, click_len, dtype=np.float32)
    # Dual formant glottal gulp: ~320Hz body and ~750Hz throat closure resonance
    body = np.sin(2 * np.pi * 320.0 * (t_c * 0.06)) * np.exp(-25.0 * t_c)
    throat = np.sin(2 * np.pi * 750.0 * (t_c * 0.06)) * np.exp(-40.0 * t_c) * 0.5
    click_sound = (body + throat) * 0.14
    wave_arr[click_pos:click_pos+click_len] = click_sound.astype(np.float32)
    return wave_arr

def generate_pause(duration_sec=0.28):
    """Natural silence pause."""
    return np.zeros(int(sr * duration_sec), dtype=np.float32)

def generate_laugh(duration_sec=0.45):
    """Conversational laughter breath burst."""
    n = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n, dtype=np.float32)
    burst_env = np.maximum(0.0, np.sin(2 * np.pi * 6.0 * t)) ** 3.0
    noise = np.random.normal(0, 0.15, n).astype(np.float32)
    vocal = 0.06 * np.sin(2 * np.pi * 190.0 * t)
    return ((noise + vocal) * burst_env * np.exp(-1.4 * t)).astype(np.float32)

# Script with paralinguistic tokens
script = "물이 계속 솟는 땅에 [breath] 조선이 궁궐을 세운 기적의 비밀 [침] 바로 땅 밑으로 거대한 석조 암거 배수로를 구축했기 때문입니다 [pause] 실로 놀랍지 않습니까"

tag_pattern = re.compile(r"(\[breath\]|\[침\]|\[pause\]|\[laugh\]|\[한숨\]|\[호흡\])", re.IGNORECASE)
tokens = tag_pattern.split(script)

print(f"\n[2/4] Parsing Script into Phonetic & Paralinguistic Blocks...")
print(f"      Full Script: \"{script}\"")

segments = []
for tok in tokens:
    tok = tok.strip()
    if not tok:
        continue
    tok_lower = tok.lower()
    if tok_lower in ("[breath]", "[호흡]"):
        print("      [Token] Inhale Breath [breath] -> Injecting organic breath wave")
        segments.append(generate_inhale_breath(0.32))
    elif tok_lower in ("[침]", "[swallow]"):
        print("      [Token] Swallow Gulp [침] -> Injecting organic gulp wave")
        segments.append(generate_swallow_gulp(0.26))
    elif tok_lower in ("[pause]", "[쉼]"):
        print("      [Token] Conversational Pause [pause] -> Injecting silence interval")
        segments.append(generate_pause(0.30))
    elif tok_lower in ("[laugh]", "[웃음]"):
        print("      [Token] Chuckle [laugh] -> Injecting laughter burst")
        segments.append(generate_laugh(0.40))
    else:
        print(f"      [Speech] Neural Synthesizing: \"{tok}\"...")
        sub_res = base_engine.synthesize(tok, speed=1.0)
        pcm = sub_res.audio_buffer.samples
        # Smooth boundary trimming
        trim_in = int(sr * 0.05)
        trim_out = int(sr * 0.05)
        if len(pcm) > (trim_in + trim_out):
            pcm = pcm[trim_in:-trim_out]
        segments.append(pcm)

# Cross-fade stitching
print("\n[3/4] Seamlessly Stitching Acoustic Waveforms with Equal-Power Crossfade...")
total_samples = sum(len(s) for s in segments)
full_audio = []

for i, seg in enumerate(segments):
    if len(full_audio) == 0:
        full_audio.extend(seg)
    else:
        # 15ms crossfade overlap
        xfade_len = int(sr * 0.015)
        if len(full_audio) >= xfade_len and len(seg) >= xfade_len:
            overlap_prev = np.array(full_audio[-xfade_len:], dtype=np.float32)
            overlap_next = np.array(seg[:xfade_len], dtype=np.float32)
            fade_out = np.cos(np.linspace(0, np.pi/2, xfade_len)) ** 2
            fade_in = np.sin(np.linspace(0, np.pi/2, xfade_len)) ** 2
            blended = overlap_prev * fade_out + overlap_next * fade_in
            full_audio[-xfade_len:] = blended.tolist()
            full_audio.extend(seg[xfade_len:])
        else:
            full_audio.extend(seg)

audio_arr = np.array(full_audio, dtype=np.float32)
peak = np.max(np.abs(audio_arr))
if peak > 0:
    audio_arr = audio_arr / peak * 0.92

duration_sec = len(audio_arr) / float(sr)
print(f"      Composite Audio: {len(audio_arr)} samples ({duration_sec:.2f}s), Peak={peak:.3f}")

out_path = Path.home() / "tts_test_s25/s25_real_expressive_shorts_10s.wav"
pcm_int16 = (audio_arr * 32767).astype(np.int16)
with wave.open(str(out_path), "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(pcm_int16.tobytes())

print(f"\n[4/4] Saved Expressive Audio: {out_path} ({out_path.stat().st_size} bytes)")
print("=" * 65)
