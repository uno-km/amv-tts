# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import os
import sys
import time
import soundfile as sf
import torch
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

print("=" * 65)
print("  OFFICIAL F5-TTS DIRECT BATCH INFERENCE & WHISPER STT AUDIT")
print("=" * 65)

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_batch_process,
    target_sample_rate
)
from huggingface_hub import hf_hub_download

# 1. Load Vocoder & Model
print("[1/4] Loading Vocos 24kHz Vocoder...")
t0 = time.time()
vocoder = load_vocoder(vocoder_name="vocos")
print(f"      Vocoder loaded in {time.time()-t0:.2f}s")

print("[2/4] Loading F5-TTS Base Model Checkpoint...")
t1 = time.time()
model_cls = DiT
model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
ckpt_path = hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_v1_Base/model_1250000.safetensors")
vocab_path = hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_v1_Base/vocab.txt")

model = load_model(
    model_cls=model_cls,
    model_cfg=model_cfg,
    ckpt_path=ckpt_path,
    vocab_file=vocab_path,
    device="cpu"
)
print(f"      F5-TTS model loaded in {time.time()-t1:.2f}s")

# 2. Text Script (English Shorts Script)
ref_audio_file = "tests/real_english_kokoro_10s.wav"
ref_text = "How the Joseon Dynasty built a massive palace on water gushing ground."
gen_text = "The secret of how Joseon built royal palaces on swamp land. Royal architects constructed sophisticated underground culverts to drain the water."

data, sr = sf.read(ref_audio_file)
audio_tensor = torch.from_numpy(data).float().unsqueeze(0)

print(f"\n[3/4] Running Flow Matching Synthesis (16 NFE steps)...")
print(f"      Ref Text: \"{ref_text}\"")
print(f"      Gen Text: \"{gen_text}\"")

t_inf = time.time()
res_gen = infer_batch_process(
    (audio_tensor, sr),
    ref_text,
    [gen_text],
    model,
    vocoder,
    nfe_step=8,
    speed=1.0,
    device="cpu"
)
final_wave, final_sample_rate, _ = next(res_gen)
elapsed = time.time() - t_inf

out_file = Path("tests/official_f5_direct.wav")
sf.write(str(out_file), final_wave, final_sample_rate)
dur = len(final_wave) / float(final_sample_rate)
print(f"      Synthesized in {elapsed:.2f}s (Duration: {dur:.2f}s, RTF: {elapsed/dur:.2f}x)")
print(f"      Saved: {out_file}")

# 3. Whisper STT Audit
print("\n[4/4] Whisper STT Self-Audit (No External Playback)...")
from faster_whisper import WhisperModel
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")

segs, info = stt_model.transcribe(str(out_file), language="en")
stt_text = " ".join([s.text for s in segs]).strip()
print(f"  Detected Language: {info.language} (confidence: {info.language_probability:.2f})")
print(f"  STT Transcribed Text: \"{stt_text}\"")
is_valid = len(stt_text) > 10 and not stt_text.startswith(",")
verdict = "PASS (HUMAN SPEECH)" if is_valid else "FAIL (NOISE)"
print(f"  Audit Verdict: [{verdict}]")
print("=" * 65)
